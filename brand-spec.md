# LuminaPV — Console EL · Especificação de marca

Fonte: brief do usuário (paleta VS Code / Dark Technical + acento #007ACC). Valores extraídos
literalmente do brief; nenhum token foi inventado.

## Tokens (tema escuro — padrão)

| Token | Hex | OKLch (aprox.) | Uso |
|---|---|---|---|
| `--bg` | `#1E1E1E` | `oklch(22.0% 0 0)` | canvas do editor / viewport |
| `--surface` | `#252526` | `oklch(25.4% 0 0)` | sidebars e painéis |
| `--surface2` | `#333333` | `oklch(32.2% 0 0)` | barras de ferramentas e cabeçalhos |
| `--fg` | `#CCCCCC` | `oklch(85.4% 0 0)` | texto de corpo |
| `--fg-strong` | `#FFFFFF` | `oklch(100% 0 0)` | títulos e valores |
| `--muted` | `#858585` | `oklch(62.7% 0 0)` | chaves/rótulos sobre `--bg` |
| `--muted2` | `#A0A0A0` | `oklch(72.4% 0 0)` | rótulos sobre `--surface` (correção de contraste ≥4,5:1) |
| `--border` | `#2D2D30` | `oklch(26.9% 0 0)` | divisores de 1px |
| `--border-strong` | `#3E3E42` | `oklch(35.0% 0 0)` | bordas de controles |
| `--accent` | `#007ACC` | `oklch(54.5% 0.140 242)` | foco, seleção, botão primário |
| `--focus` | `#4FA6E0` | `oklch(68.6% 0.113 242)` | anel de foco (≥3:1 contra superfícies escuras) |
| `--defect` | `#F44747` | `oklch(62.4% 0.204 25)` | célula defeituosa / falha (preenchimento e bordas) |
| `--defect-ink` | `#FF6161` | `oklch(69.8% 0.204 25)` | texto vermelho sobre superfície escura (≥4,5:1 em `--bg`, `--surface`, `--surface2`) |
| `--pass` | `#89D185` | `oklch(81.6% 0.119 141)` | aprovado |
| `--pass2` | `#4EC9B0` | `oklch(76.4% 0.088 189)` | nominal (teal de token) |
| `--warn` | `#DCDCAA` | `oklch(87.0% 0.075 106)` | atenção |
| `--status` | `#005FA3` | `oklch(45.3% 0.120 242)` | barra de status (texto branco ≥6:1) |

Tema claro (`body.theme-light`) re-deriva os mesmos papéis a partir de VS Code Light+:
`#FFFFFF / #F3F3F3 / #E7E7E7 / #333333 / #000000 / #6B6B6B / #CECECE / #005FB8 / #ADD6FF /
#CD3131 / #107C10 / #0E7A6B / #8A6D00`.

## Tipografia

- Interface: `"Segoe UI", "Inter", system-ui, -apple-system, sans-serif` — 11–13px, tracking normal.
- Telemetria (dimensões, coordenadas, índices de célula, percentuais): `"JetBrains Mono", "Consolas", "SF Mono", ui-monospace, monospace`.
- Rótulos de seção: 10px maiúsculas, tracking `0.08em`, `--muted2`.

## Regras observadas do sistema

1. Raio `0` por padrão; `2px` apenas em botões e tags de status. Nenhum pílula, nenhum blur, nenhuma sombra decorativa.
2. Toda separação entre painéis operacionais é `1px solid var(--border)` — nenhum espaçamento faz o papel de divisor.
3. Uma única ação primária preenchida com `--accent` por viewport; todo o resto é contorno, fantasma ou link de texto.
4. Densidade de instrumento: linhas de tabela em 24px, altura de toolbar 40px, barra de status 22px.
5. Feedback de interação move luminância de fundo (L ±0,06–0,12) ou espessura de borda; a cor do texto nunca se aproxima do fundo em hover.
6. Vermelho e teal codificam estado físico (defeito/nominal), nunca decoração. O acento azul é reservado para foco, seleção e a ação primária.
7. `--defect` só é usado como preenchimento, borda ou ícone; qualquer **texto** vermelho usa `--defect-ink`, porque `#F44747` fica em 4,2:1 sobre superfícies de painel (abaixo de 4,5:1). Tags de status usam o fundo `--bg` como pastilha, garantindo contraste sobre `--surface2`.
8. A única ação primária preenchida é `Ingerir lote`, na barra superior; os botões dos painéis de estado (vazio, erro, nova tentativa) são de contorno.
9. A doca inferior tem altura fixa (232px) com rolagem interna, para que payloads longos nunca comprimam o viewport; o cabeçalho permanece visível quando recolhida (28px).

## Frase de sistema

Workstation IDE industrial: superfícies planas de 1px, geometria 90°, telemetria em monoespaçada de alto contraste e o azul de acento reservado exclusivamente para foco e seleção.
