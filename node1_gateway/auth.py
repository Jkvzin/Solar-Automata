"""
LuminaPV Authentication & Identity Module (Node 1 Gateway)
==========================================================
Implements:
- Cryptographic password hashing (PBKDF2-HMAC-SHA256, 260,000 iterations, constant-time verification)
- Stateless RFC 7519 JWT (HS256) generation, signature verification, and JTI revocation blacklist
- Hexagonal Architecture (Ports & Adapters / Repository Pattern) for User Persistence:
  - UserRepository (Abstract Port)
  - InMemoryUserRepository (Local/Test Mock Adapter with pre-seeded operator account)
  - S3UserRepository (Amazon S3 Adapter with automatic fallback until S3 IAM/Bucket setup is completed)
- FastAPI Router exposing /api/auth/* and /api/v1/auth/* endpoints
"""

from abc import ABC, abstractmethod
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

logger = logging.getLogger("node1_gateway.auth")

# -----------------------------------------------------------------------------
# Security & Storage Configuration
# -----------------------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "luminapv-industrial-secret-key-2026-hs256-solar-automata")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_SECONDS = int(os.getenv("JWT_EXPIRATION_SECONDS", "28800"))  # 8h operational shift
PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "260000"))

AUTH_STORAGE_BACKEND = os.getenv("AUTH_STORAGE_BACKEND", "memory").lower()  # "memory" | "s3"
AUTH_S3_BUCKET = os.getenv(
    "AUTH_S3_BUCKET",
    os.getenv("AWS_S3_BUCKET", "dataset-paineis-solares-joaoguilherme-joaovictor-gustavo-2026")
)
AUTH_S3_PREFIX = os.getenv("AUTH_S3_PREFIX", "auth").strip("/")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))
AUTH_ENFORCE_PROTECTED_ROUTES = os.getenv("AUTH_ENFORCE_PROTECTED_ROUTES", "false").lower() in ("true", "1", "yes")

DEFAULT_OPERATOR_EMAIL = os.getenv("DEFAULT_OPERATOR_EMAIL", "operador@luminapv.local")
DEFAULT_OPERATOR_PASSWORD = os.getenv("DEFAULT_OPERATOR_PASSWORD", "LuminaPV@2026")
DEFAULT_OPERATOR_NAME = os.getenv("DEFAULT_OPERATOR_NAME", "Operador Chefe LuminaPV")

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
ALLOWED_ROLES = {"operator", "engineer", "auditor", "admin", "viewer"}


# -----------------------------------------------------------------------------
# Domain Models & Data Transfer Objects (DTOs)
# -----------------------------------------------------------------------------
class UserRecord(BaseModel):
    """Canonical internal user entity persisted in the repository (S3 / Memory)."""
    user_id: str = Field(..., description="Unique immutable UUIDv4 identifier (e.g. usr-...).")
    email: str = Field(..., description="Normalized lowercase email address.")
    name: str = Field(..., description="Full name of the operator or engineer.")
    password_hash: str = Field(..., description="PBKDF2-HMAC-SHA256 encoded password hash.")
    role: str = Field(default="operator", description="Access role: operator | admin | viewer.")
    is_active: bool = Field(default=True, description="Whether the operator account is active.")
    created_at: str = Field(..., description="ISO 8601 UTC creation timestamp.")
    updated_at: str = Field(..., description="ISO 8601 UTC last update timestamp.")
    last_login_at: Optional[str] = Field(default=None, description="ISO 8601 UTC last login timestamp.")


class UserPublicDTO(BaseModel):
    """Sanitized public user representation returned to the frontend (never exposes password_hash)."""
    user_id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: str
    last_login_at: Optional[str] = None
    storage_backend: str = "memory"


class RegisterRequest(BaseModel):
    """Payload contract for POST /api/auth/register."""
    name: str = Field(..., min_length=2, max_length=120, description="Full operator name")
    email: str = Field(..., min_length=5, max_length=180, description="Corporate or local email")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 characters)")
    role: Optional[str] = Field(default="operator", description="Role: operator | admin | viewer")


class LoginRequest(BaseModel):
    """Payload contract for POST /api/auth/login."""
    email: str = Field(..., min_length=3, max_length=180)
    password: str = Field(..., min_length=1, max_length=128)


class AuthTokenResponse(BaseModel):
    """Standardized authentication response containing JWT access token and operator profile."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    expires_at: str
    user: UserPublicDTO


# -----------------------------------------------------------------------------
# Cryptographic Utilities: Password Hashing & JWT (HS256)
# -----------------------------------------------------------------------------
def hash_password(plain_password: str, salt: Optional[str] = None) -> str:
    """
    Hashes a plaintext password using PBKDF2-HMAC-SHA256 with a 128-bit cryptographic salt.
    Output format: pbkdf2_sha256$<iterations>$<salt_hex>$<digest_hex>
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    salt_hex = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt_hex.encode("utf-8"),
        PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_hex}${dk.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verifies a plaintext password against a stored PBKDF2-HMAC-SHA256 hash in constant time.
    """
    if not plain_password or not stored_hash:
        return False
    try:
        parts = stored_hash.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt_hex = parts[2]
        expected_hex = parts[3]
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt_hex.encode("utf-8"),
            iterations
        )
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    rem = len(segment) % 4
    if rem > 0:
        segment += "=" * (4 - rem)
    return base64.urlsafe_b64decode(segment.encode("ascii"))


# In-memory revoked token JTI store (maps jti -> exp timestamp)
_REVOKED_TOKENS: Dict[str, int] = {}


def create_access_token(user: UserRecord, expires_delta: int = JWT_EXPIRATION_SECONDS) -> Tuple[str, int, str, str]:
    """
    Generates an RFC 7519 HS256 JWT token for the authenticated user.
    Returns: (jwt_token, expires_in_seconds, expires_at_iso, jti)
    """
    now_ts = int(time.time())
    exp_ts = now_ts + expires_delta
    jti = str(uuid.uuid4())
    expires_at_iso = datetime.fromtimestamp(exp_ts, tz=timezone.utc).isoformat()

    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": user.user_id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "iat": now_ts,
        "exp": exp_ts,
        "jti": jti,
        "iss": "luminapv-gateway"
    }

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    token = f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"

    return token, expires_delta, expires_at_iso, jti


def revoke_token(jti: str, exp: int) -> None:
    """Adds a token JTI to the revocation blacklist and prunes expired entries."""
    now_ts = int(time.time())
    expired = [k for k, v in _REVOKED_TOKENS.items() if v < now_ts]
    for k in expired:
        _REVOKED_TOKENS.pop(k, None)
    _REVOKED_TOKENS[jti] = exp


def decode_and_verify_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates an HS256 JWT token, checking signature integrity, expiration, and revocation.
    Raises HTTPException(401) if invalid, expired, or revoked.
    """
    if not token or token.count(".") != 2:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticação malformado ou ausente.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    header_b64, payload_b64, sig_b64 = token.split(".")
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()

    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Assinatura do token inválida.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Assinatura criptográfica do token inválida.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Payload do token inválido.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    now_ts = int(time.time())
    exp = int(payload.get("exp", 0))
    if exp <= now_ts:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessão expirada. Faça login novamente.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    jti = payload.get("jti")
    if jti and jti in _REVOKED_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token revogado por logout.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return payload


# -----------------------------------------------------------------------------
# Ports & Adapters: User Repository Interface & Implementations
# -----------------------------------------------------------------------------
class UserRepository(ABC):
    """
    Abstract Port defining the persistence contract for User identity records.
    Allows seamless switching between InMemoryUserRepository (local/testing)
    and S3UserRepository (production Amazon S3 bucket storage).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Returns the identifier of the active storage backend ('memory' or 's3')."""

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Retrieves a user record by its unique user_id, or None if not found."""

    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Retrieves a user record by normalized email address, or None if not found."""

    @abstractmethod
    async def create_user(self, user: UserRecord) -> UserRecord:
        """Persists a new user record and updates the email lookup index."""

    @abstractmethod
    async def update_user(self, user: UserRecord) -> UserRecord:
        """Updates an existing user record (e.g., last_login_at, role, password_hash)."""

    @abstractmethod
    async def list_users(self) -> List[UserRecord]:
        """Returns all registered user records."""


class InMemoryUserRepository(UserRepository):
    """
    Thread/Async-safe In-Memory Adapter implementing UserRepository.
    Pre-seeded with a default operator account so local development and CI tests
    work out of the box without requiring live AWS S3 credentials.
    """

    def __init__(self, seed_default: bool = True) -> None:
        self._users_by_id: Dict[str, UserRecord] = {}
        self._email_to_id: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        if seed_default:
            self._seed_default_operator()

    def _seed_default_operator(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        email_norm = DEFAULT_OPERATOR_EMAIL.strip().lower()
        default_user = UserRecord(
            user_id="usr-default-operator-01",
            email=email_norm,
            name=DEFAULT_OPERATOR_NAME,
            password_hash=hash_password(DEFAULT_OPERATOR_PASSWORD, salt="a1b2c3d4e5f60718293a4b5c6d7e8f90"),
            role="admin",
            is_active=True,
            created_at=now_iso,
            updated_at=now_iso,
            last_login_at=None
        )
        self._users_by_id[default_user.user_id] = default_user
        self._email_to_id[email_norm] = default_user.user_id

    @property
    def backend_name(self) -> str:
        return "memory"

    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        async with self._lock:
            u = self._users_by_id.get(user_id)
            return u.model_copy(deep=True) if u else None

    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        email_norm = email.strip().lower()
        async with self._lock:
            uid = self._email_to_id.get(email_norm)
            if not uid:
                return None
            u = self._users_by_id.get(uid)
            return u.model_copy(deep=True) if u else None

    async def create_user(self, user: UserRecord) -> UserRecord:
        email_norm = user.email.strip().lower()
        async with self._lock:
            if email_norm in self._email_to_id:
                raise ValueError(f"Email '{email_norm}' already registered.")
            record = user.model_copy(update={"email": email_norm})
            self._users_by_id[record.user_id] = record
            self._email_to_id[email_norm] = record.user_id
            return record.model_copy(deep=True)

    async def update_user(self, user: UserRecord) -> UserRecord:
        email_norm = user.email.strip().lower()
        async with self._lock:
            record = user.model_copy(
                update={
                    "email": email_norm,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            )
            self._users_by_id[record.user_id] = record
            self._email_to_id[email_norm] = record.user_id
            return record.model_copy(deep=True)

    async def list_users(self) -> List[UserRecord]:
        async with self._lock:
            return [u.model_copy(deep=True) for u in self._users_by_id.values()]


class S3UserRepository(UserRepository):
    """
    Amazon S3 Adapter for UserRepository.
    Object Layout Contract (for AWS S3 Infrastructure Integration):
      - User Record: s3://<bucket>/<prefix>/users/by-id/<user_id>.json
      - Email Index: s3://<bucket>/<prefix>/users/by-email/<email_sha256>.json
    Includes automatic graceful fallback to InMemoryUserRepository when S3 credentials
    or bucket policies are not yet active in the local environment.
    """

    def __init__(
        self,
        bucket_name: str = AUTH_S3_BUCKET,
        prefix: str = AUTH_S3_PREFIX,
        s3_client: Optional[Any] = None,
        fallback_repo: Optional[InMemoryUserRepository] = None
    ) -> None:
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.s3_client = s3_client
        self.fallback = fallback_repo or InMemoryUserRepository(seed_default=True)

    @property
    def backend_name(self) -> str:
        return "s3" if self.s3_client is not None else "s3-fallback-memory"

    def _user_key(self, user_id: str) -> str:
        return f"{self.prefix}/users/by-id/{user_id}.json"

    def _email_index_key(self, email: str) -> str:
        email_norm = email.strip().lower()
        email_hash = hashlib.sha256(email_norm.encode("utf-8")).hexdigest()
        return f"{self.prefix}/users/by-email/{email_hash}.json"

    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        if self.s3_client is not None:
            try:
                resp = await asyncio.to_thread(
                    self.s3_client.get_object,
                    Bucket=self.bucket_name,
                    Key=self._user_key(user_id)
                )
                raw = resp["Body"].read().decode("utf-8")
                return UserRecord(**json.loads(raw))
            except Exception as exc:
                logger.debug(f"S3 get_user_by_id('{user_id}') fallback: {exc}")
        return await self.fallback.get_user_by_id(user_id)

    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        email_norm = email.strip().lower()
        if self.s3_client is not None:
            try:
                idx_resp = await asyncio.to_thread(
                    self.s3_client.get_object,
                    Bucket=self.bucket_name,
                    Key=self._email_index_key(email_norm)
                )
                idx_data = json.loads(idx_resp["Body"].read().decode("utf-8"))
                user_id = idx_data.get("user_id")
                if user_id:
                    return await self.get_user_by_id(user_id)
            except Exception as exc:
                logger.debug(f"S3 get_user_by_email('{email_norm}') fallback: {exc}")
        return await self.fallback.get_user_by_email(email_norm)

    async def create_user(self, user: UserRecord) -> UserRecord:
        # Always keep local fallback synchronized for read-your-writes consistency
        created = await self.fallback.create_user(user)
        if self.s3_client is not None:
            try:
                user_bytes = json.dumps(created.model_dump(), indent=2).encode("utf-8")
                index_payload = {
                    "user_id": created.user_id,
                    "email": created.email,
                    "updated_at": created.updated_at
                }
                index_bytes = json.dumps(index_payload, indent=2).encode("utf-8")

                await asyncio.to_thread(
                    self.s3_client.put_object,
                    Bucket=self.bucket_name,
                    Key=self._user_key(created.user_id),
                    Body=user_bytes,
                    ContentType="application/json"
                )
                await asyncio.to_thread(
                    self.s3_client.put_object,
                    Bucket=self.bucket_name,
                    Key=self._email_index_key(created.email),
                    Body=index_bytes,
                    ContentType="application/json"
                )
            except Exception as exc:
                logger.warning(f"S3 create_user persistence fallback (stored in memory): {exc}")
        return created

    async def update_user(self, user: UserRecord) -> UserRecord:
        updated = await self.fallback.update_user(user)
        if self.s3_client is not None:
            try:
                user_bytes = json.dumps(updated.model_dump(), indent=2).encode("utf-8")
                await asyncio.to_thread(
                    self.s3_client.put_object,
                    Bucket=self.bucket_name,
                    Key=self._user_key(updated.user_id),
                    Body=user_bytes,
                    ContentType="application/json"
                )
            except Exception as exc:
                logger.warning(f"S3 update_user persistence fallback: {exc}")
        return updated

    async def list_users(self) -> List[UserRecord]:
        return await self.fallback.list_users()


# Initialize Singleton Repository Instance
_memory_repo = InMemoryUserRepository(seed_default=True)
if AUTH_STORAGE_BACKEND == "s3":
    try:
        import boto3
        _s3_auth_client = boto3.client("s3", region_name=AWS_REGION)
    except Exception:
        _s3_auth_client = None
    user_repository: UserRepository = S3UserRepository(
        bucket_name=AUTH_S3_BUCKET,
        prefix=AUTH_S3_PREFIX,
        s3_client=_s3_auth_client,
        fallback_repo=_memory_repo
    )
else:
    user_repository = _memory_repo


def get_user_repository() -> UserRepository:
    return user_repository


def to_public_dto(user: UserRecord, repo: UserRepository) -> UserPublicDTO:
    return UserPublicDTO(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        storage_backend=repo.backend_name
    )


# -----------------------------------------------------------------------------
# FastAPI Dependencies for Route Protection
# -----------------------------------------------------------------------------
def _extract_bearer_token(authorization: Optional[str], request: Optional[Request] = None) -> Optional[str]:
    if authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    repo: UserRepository = Depends(get_user_repository)
) -> UserRecord:
    """
    Mandatory authentication dependency.
    Verifies Bearer JWT token and resolves the active UserRecord from UserRepository.
    """
    token = _extract_bearer_token(authorization, request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticação necessária. Envie o cabeçalho Authorization: Bearer <token>.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = decode_and_verify_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido: identificador de usuário ausente.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = await repo.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Operador não encontrado ou conta desativada.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


# Alias for standard FastAPI dependency naming convention
get_current_active_user = get_current_user


async def verify_optional_or_enforced_auth(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    repo: UserRepository = Depends(get_user_repository)
) -> Optional[UserRecord]:
    """
    Validates Bearer token if provided, or enforces authentication when
    AUTH_ENFORCE_PROTECTED_ROUTES=true.
    """
    token = _extract_bearer_token(authorization, request)
    if not token:
        if AUTH_ENFORCE_PROTECTED_ROUTES:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Acesso negado: autenticação obrigatória para enviar lotes de inspeção.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return None
    return await get_current_user(request, authorization, repo)


# -----------------------------------------------------------------------------
# FastAPI Authentication Router (/api/auth/* and /api/v1/auth/*)
# -----------------------------------------------------------------------------
auth_router = APIRouter(tags=["Authentication"])


@auth_router.post(
    "/api/auth/register",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new LuminaPV operator account"
)
@auth_router.post(
    "/api/v1/auth/register",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False
)
async def register_operator(
    payload: RegisterRequest,
    response: Response,
    repo: UserRepository = Depends(get_user_repository)
):
    email_norm = payload.email.strip().lower()
    if not EMAIL_REGEX.match(email_norm):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Formato de e-mail inválido."
        )

    name_clean = payload.name.strip()
    if len(name_clean) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="O nome do operador deve conter pelo menos 2 caracteres."
        )

    if len(payload.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A senha deve possuir no mínimo 8 caracteres."
        )

    role_norm = (payload.role or "operator").strip().lower()
    if role_norm not in ALLOWED_ROLES:
        role_norm = "operator"

    existing = await repo.get_user_by_email(email_norm)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"O e-mail '{email_norm}' já está cadastrado no sistema."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    new_user = UserRecord(
        user_id=f"usr-{uuid.uuid4().hex[:12]}",
        email=email_norm,
        name=name_clean,
        password_hash=hash_password(payload.password),
        role=role_norm,
        is_active=True,
        created_at=now_iso,
        updated_at=now_iso,
        last_login_at=now_iso
    )

    created = await repo.create_user(new_user)
    token, expires_in, expires_at, _ = create_access_token(created)

    response.set_cookie(
        key="luminapv_token",
        value=token,
        httponly=True,
        max_age=expires_in,
        samesite="lax"
    )

    logger.info(f"Novo operador registrado: {created.email} (id={created.user_id}, role={created.role})")
    return AuthTokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=expires_in,
        expires_at=expires_at,
        user=to_public_dto(created, repo)
    )


@auth_router.post(
    "/api/auth/login",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate operator credentials and issue JWT"
)
@auth_router.post(
    "/api/v1/auth/login",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def login_operator(
    payload: LoginRequest,
    response: Response,
    repo: UserRepository = Depends(get_user_repository)
):
    email_norm = payload.email.strip().lower()
    user = await repo.get_user_by_email(email_norm)

    if user is None or not verify_password(payload.password, user.password_hash):
        logger.warning(f"Tentativa de login inválida para: '{email_norm}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas. Verifique seu e-mail e senha.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conta de operador desativada. Contate o administrador."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    user.last_login_at = now_iso
    updated = await repo.update_user(user)

    token, expires_in, expires_at, _ = create_access_token(updated)
    response.set_cookie(
        key="luminapv_token",
        value=token,
        httponly=True,
        max_age=expires_in,
        samesite="lax"
    )

    logger.info(f"Login efetuado com sucesso: {updated.email} (id={updated.user_id})")
    return AuthTokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=expires_in,
        expires_at=expires_at,
        user=to_public_dto(updated, repo)
    )


@auth_router.post(
    "/api/auth/logout",
    status_code=status.HTTP_200_OK,
    summary="Revoke active JWT token and terminate operator session"
)
@auth_router.post(
    "/api/v1/auth/logout",
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def logout_operator(
    request: Request,
    response: Response,
    authorization: Optional[str] = Header(default=None)
):
    token = _extract_bearer_token(authorization, request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nenhum token de sessão ativo fornecido para logout.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = decode_and_verify_token(token)
    jti = payload.get("jti")
    exp = int(payload.get("exp", int(time.time()) + 3600))
    if jti:
        revoke_token(jti, exp)

    response.delete_cookie("luminapv_token")
    logger.info(f"Sessão encerrada para operador: {payload.get('email', 'unknown')} (jti={jti})")
    return {
        "status": "logged_out",
        "message": "Sessão encerrada e token revogado com segurança.",
        "revoked_jti": jti
    }


@auth_router.get(
    "/api/auth/me",
    response_model=UserPublicDTO,
    status_code=status.HTTP_200_OK,
    summary="Return currently authenticated operator profile"
)
@auth_router.get(
    "/api/v1/auth/me",
    response_model=UserPublicDTO,
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def get_authenticated_operator(
    current_user: UserRecord = Depends(get_current_user),
    repo: UserRepository = Depends(get_user_repository)
):
    return to_public_dto(current_user, repo)


@auth_router.get(
    "/api/auth/status",
    status_code=status.HTTP_200_OK,
    summary="Return authentication subsystem & S3 repository adapter status"
)
@auth_router.get(
    "/api/v1/auth/status",
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def get_auth_subsystem_status(repo: UserRepository = Depends(get_user_repository)):
    users = await repo.list_users()
    return {
        "status": "healthy",
        "storage_backend": repo.backend_name,
        "s3_bucket_target": AUTH_S3_BUCKET,
        "s3_key_prefix": AUTH_S3_PREFIX,
        "jwt_algorithm": JWT_ALGORITHM,
        "password_kdf": f"PBKDF2-HMAC-SHA256 ({PBKDF2_ITERATIONS} iterations)",
        "enforce_protected_routes": AUTH_ENFORCE_PROTECTED_ROUTES,
        "registered_operators_count": len(users)
    }
