## Descrição das Alterações

<!-- Descreva de forma clara e objetiva o que foi alterado nesta branch -->

- [x] Integração do modelo de classificação YOLOv8-cls (`best.pt`) para diagnóstico de células solares individuais.
- [x] Detecção dinâmica de grade e recorte de células em memória sem I/O em disco.
- [x] Anotações visuais com destaque translúcido e retângulos vermelhos nas células defeituosas com percentual de confiança.
- [x] Cálculo consolidado da porcentagem de dano do módulo solar e verificação PASS / FAIL (limite 5.0%).
- [x] Script CLI de teste `test_panel.py` e atualização da suíte de testes unitários e de integração.

## Revisor Designado
- [ ] @JoaoGarciaM (João Guilherme)

## Checklist
- [x] Testes unitários executados e aprovados (`test_cv_engine.py`)
- [x] Testes de integração da API executados e aprovados (`test_api_integration.py`)
- [x] Validação ponta a ponta com `test_panel.py` gerando imagem anotada
- [x] `contexto.md` protegido pelo `.gitignore`
