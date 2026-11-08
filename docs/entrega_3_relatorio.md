# Relatório Técnico — Entrega 3: Arquitetura de Sistemas Distribuídos (LuminaPV / Solar-Automata)

**Repositório:** `https://github.com/Jkvzin/Solar-Automata`  
**Status da Entrega:** Concluída, Integrada na `main` e Validada (`43/43` testes automatizados aprovados)

---

## 1. Visão Geral da Arquitetura Distribuída

A plataforma **Solar-Automata (LuminaPV)** realiza a inspeção óptica automatizada de módulos fotovoltaicos por imagens de Eletroluminescência (EL), combinando retificação homográfica robusta, segmentação adaptativa de células (*Full-Cell* e *Half-Cut*) e classificação neural em lote com **YOLOv8-cls** (`best.pt`).

Na **Entrega 3**, a plataforma foi evoluída para um cluster distribuído tolerante a falhas, contemplando formalmente os três pilares de Sistemas Distribuídos:
1. **Mecanismos de Coordenação:** Eleição de Líder baseada em concessão (*lease*) distribuída e Exclusão Mútua Distribuída atômica via script Lua.
2. **Esquema de Nomeação e Descoberta de Serviços:** Domínio DNS estruturado (`.luminapv.local`), catálogo dinâmico com *heartbeat* e roteamento *Round-Robin* com *failover* automático.
3. **Replicação e Consistência:** Replicação *Primary-Backup* de resultados e cache distribuído de inferência com verificação criptográfica de integridade do modelo (`model_sha256`), operando sob **Modelo de Consistência Híbrido** (*Read-Your-Writes* + *Consistência Eventual*).

---

## 2. Detalhamento dos Três Pilares da Entrega 3

### 2.1. Mecanismos de Coordenação (`node2_worker/main.py` & `node1_gateway/main.py`)

#### A. Eleição de Líder (`LeaderElectionManager`)
- **Algoritmo:** Disputa de concessão (*lease*) atômica no Redis via comando `SET luminapv:leader <payload_json> NX EX 8`.
- **Ciclo de Vida e Renovação:**
  - O worker vencedor incrementa monotonicamente o termo da eleição (`INCR luminapv:leader:term`), assume o papel `LEADER` e renova o lease a cada `2.0s` (`ELECTION_CHECK_INTERVAL_SECONDS`).
  - As demais réplicas identificam o líder ativo e operam como `FOLLOWER`.
  - **Failover Automático:** Caso o worker líder sofra falha ou parada (`docker stop`), o lease expira em `8s` (`LEADER_LEASE_TTL_SECONDS`) e a réplica seguidora assume a liderança no próximo ciclo de verificação, sem intervenção manual.
  - **Encerramento Gracioso e Simulação:** No `stop()` ou via `POST /api/v1/cluster/election?action=step_down`, o líder libera imediatamente a chave `luminapv:leader`, permitindo demonstrar reeleições instantâneas.

#### B. Exclusão Mútua Distribuída (`DistributedLock` & `DistributedLockManager`)
- **Proteção da Seção Crítica:** No processamento em lote (`POST /api/v1/analyze/bulk`), cada imagem recebida tem seu hash `content_sha256` calculado e adquire um lock distribuído exclusivo `lock:module:<content_sha256>` (`SET ... NX EX 30`).
- **Prevenção de Deadlock e Liberação Segura (Script Lua):**
  - Todo lock possui `TTL = 30s` para evitar *deadlocks* caso um gateway caia durante o processamento.
  - A liberação ocorre em bloco `finally` executando o script atômico `_LUA_RELEASE_LOCK` no Redis, que verifica se o `token` pertence ao proprietário antes de remover a chave e limpar o registro em `luminapv:active_locks`.

---

### 2.2. Esquema de Nomeação e Roteamento (`docker-compose.yml`, `node2_worker` & `node1_gateway`)

#### A. Endereçamento DNS Estruturado e Identidade Lógica
- **Nome Lógico do Serviço de Visão:** `service://luminapv-vision-worker`
- **Domínio DNS da Rede Bridge (`luminapv-network`):**
  - `gateway.luminapv.local` (`node1_gateway` — Porta `8080`)
  - `state.luminapv.local` e `redis.luminapv.local` (`luminapv-registry` — Porta `6379`)
  - `worker-1.luminapv.local` (`node2-worker-1` — Porta `8001`)
  - `worker-2.luminapv.local` (`node2-worker-2` — Porta externa `8002` / interna `8001`)

#### B. Auto-Registro Periódico (`WorkerServiceRegistry`)
- Cada worker registra e atualiza seu *heartbeat* a cada `3.0s` na chave `luminapv:services:vision-worker:<node_id>` com `TTL = 10s` e adiciona seu `node_id` ao conjunto `luminapv:services:vision-worker:nodes`.
- Durante o encerramento gracioso (`stop()`), o worker remove sua chave e executa `SREM luminapv:services:vision-worker:nodes <node_id>`, evitando referências órfãs.

#### C. Resolução Dinâmica, Balanceamento Round-Robin e Failover (`ServiceCatalogResolver`)
- O `node1_gateway` resolve dinamicamente `service://luminapv-vision-worker` consultando os nós saudáveis registrados no Redis.
- As requisições de inspeção são distribuídas em **Round-Robin** entre os workers ativos.
- Se o worker primário selecionado para um módulo apresentar falha de conexão ou timeout (`httpx.RequestError`), `dispatch_module_to_worker` executa **failover imediato** redirecionando a carga para a réplica saudável seguinte.

---

### 2.3. Replicação e Modelo de Consistência (`DistributedResultReplicator`)

#### A. Replicação Primary-Backup e Integridade do Modelo (`model_sha256`)
- **Replicação Primário-Backup:** Cada `ModuleResult` gerado é persistido sincronamente no store primário Redis (`luminapv:replica:module:<content_sha256>:yolo=<0|1>`, `TTL = 3600s`) e no cache local do worker (`LRUReplicaStore` com limite configurável `MAX_LOCAL_REPLICAS = 500`).
- **Prevenção de *Model Drift*:** O hash SHA-256 do arquivo de pesos `node2_worker/models/best.pt` (`get_model_sha256()`) é acoplado a cada resultado replicado. Antes de servir um *cache hit*, o worker verifica `cached.model_sha256 == current_model_hash`; caso os pesos tenham sido atualizados, o cache antigo é invalidado automaticamente.
- **Resiliência de Schema e Higiene de Índice:** Mudanças no contrato Pydantic de `ModuleResult` degradam de forma segura para *cache miss* (`try/except` na desserialização), e chaves expiradas por TTL são expurgadas automaticamente do conjunto `luminapv:replica:modules:index` ao consultar `GET /api/v1/replicas/stats`.

#### B. Especificação Formal do Modelo de Consistência (CAP / PACELC)
1. **Consistência Centrada no Cliente — *Read-Your-Writes* (RYW):**
   - A gravação no store primário ocorre antes do retorno HTTP do `POST /api/v1/process-module` e do `POST /api/v1/analyze/bulk`. Qualquer consulta subsequente do mesmo operador ou do console (`GET /api/v1/replicas/{sha256}`, histórico do operador ou reenvio da imagem) lê garantidamente o resultado recém-gravado.
2. **Consistência Centrada em Dados entre Réplicas — *Consistência Eventual*:**
   - Os stores locais em memória (`_LOCAL_REPLICA_STORE`) das demais réplicas de worker sincronizam os artefatos sob demanda ao consultar o store primário.
3. **Classificação CAP / PACELC (`PA/EL`):**
   - Em caso de partição ou falha de nós (**P**), o sistema prioriza **Disponibilidade (A)** via failover entre workers e fallback local em memória; em operação normal (**E**), prioriza **Baixa Latência (L)** servindo inferências em lote vetorizadas e cache distribuído sem bloqueio global entre réplicas.

---

## 3. Rastreabilidade de Issues e Pull Requests Entregues

| Issue / PR | Escopo Técnico | Responsável | Status |
| :--- | :--- | :--- | :--- |
| **Issue #1 / PR #1** | Integração do modelo classificador YOLOv8-cls (`best.pt`), anotação visual e CLI `test_panel.py` | `@Jkvzin` (João Victor) | `MERGED` |
| **Issue #2 / PR #8** | Orquestração Multi-Worker (`docker-compose.yml`), DNS `.luminapv.local` e `WorkerServiceRegistry` | `@Jkvzin` (João Victor) | `MERGED` |
| **Issue #3 / PR #11** | Eleição de Líder (`LeaderElectionManager`), Exclusão Mútua (`DistributedLock` Lua) e Resiliência AWS | `@JoaoGarciaM` (João Guilherme) | `MERGED` |
| **Issue #4 / PR #12** | Resolução Dinâmica (`ServiceCatalogResolver`), Balanceamento Round-Robin, Failover e `/cluster/status` | `@JoaoGarciaM` (João Guilherme) | `MERGED` |
| **Issue #5 / PR #9** | Replicação Primary-Backup, Consistência *Read-Your-Writes*, Inferência Batch YOLO e Doc CAP/PACELC | `@Jkvzin` (João Victor) | `MERGED` |
| **Issue #6 / PR #13** | Painel de Topologia do Cluster no Frontend, Visualização de Líder/Réplicas e Mapeamento `defeituosa` | `@GrouwBer` (Gustavo Grouwber) | `MERGED` |
| **Issue #7 / PR #14** | Polling em Tempo Real (`4s`), Detecção Visual de Failover, Feedback de Locks e Status de Desconexão | `@GrouwBer` (Gustavo Grouwber) | `MERGED` |
| **Issue #15 / PR #16** | Sistema Completo de Autenticação de Operadores (JWT HS256, PBKDF2-SHA256 & S3/Memory Adapter) | `@GrouwBer` (Gustavo Grouwber) | `MERGED` |
| **Issue #18 / PR #19** | Histórico de Análises por Operador, Galeria S3 com Pre-Signed URLs e Recarga Interativa no Inspetor | `@GrouwBer` (Gustavo Grouwber) | `MERGED` |
| **Issue #20 / PR #21** | Refatoração de Pré-processamento EL (CLAHE, Homografia Convex Hull 4 Retas e Segmentação Half-Cut) | `@GrouwBer` (Gustavo Grouwber) | `MERGED` |
| **Refinamento / PR #22** | Refinamentos de Code Review: `LRUReplicaStore`, fallback de schema, limpeza `SREM` no `stop()` e higiene Redis | `@Jkvzin` (João Victor) | `MERGED` |

---

## 4. Guia Rápido de Execução (Runbook)

### 4.1. Subindo o Cluster Completo via Docker Compose
Com um único comando na raiz do repositório:

```bash
docker compose up --build -d
```

Verifique o estado dos 4 containers:
```bash
docker compose ps
```

- **Console Web Operacional (Gateway):** `http://localhost:8080`
- **Worker 1 (API & Health):** `http://localhost:8001/health`
- **Worker 2 (API & Health):** `http://localhost:8002/health`
- **Telemetria Consolidada do Cluster:** `http://localhost:8080/api/v1/cluster/status`

### 4.2. Credenciais Padrão de Operador
No console web (`http://localhost:8080`), clique no botão de sessão do operador no topo direito:
- **E-mail:** `operador@luminapv.local`
- **Senha:** `LuminaPV@2026`

### 4.3. Testando Eleição de Líder e Failover em Tempo Real
```bash
# Consultar o estado atual do cluster e identificar o líder
curl http://localhost:8080/api/v1/cluster/status

# Simular renúncia do líder atual (step_down) para observar reeleição no painel Topologia do Cluster
curl -X POST "http://localhost:8080/api/v1/cluster/election?action=step_down"
```

### 4.4. Executando a Suíte Completa de Testes Automatizados (43 Testes)
```bash
# Executar todos os testes unitários (Visão Computacional + YOLO) e de integração distribuída
python -m unittest discover -s tests -v
```
