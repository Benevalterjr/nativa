# Política de Segurança — Linguagem Nativa

Levamos a segurança do compilador, do runtime e do ecossistema da Linguagem Nativa a sério. Agradecemos o esforço e a colaboração da comunidade em reportar vulnerabilidades de forma responsável.

## Versões Suportadas

Atualmente, apenas a versão mais recente em desenvolvimento recebe atualizações e correções de segurança.

| Versão | Suportada          |
| ------ | ------------------ |
| 0.3.x  | :white_check_mark: |
| < 0.3  | :x:                |

## Como Reportar uma Vulnerabilidade

Se você descobrir uma falha de segurança no código da Nativa, no compilador AOT C99, no interpretador ou nas dependências:

1. **Não abra uma Issue pública** para não expor a falha antes da correção.
2. Utilize o recurso nativo de **[Private Vulnerability Reporting do GitHub](https://github.com/Benevalterjr/nativa/security/advisories/new)** para relatar a falha diretamente aos mantenedores.
3. Se preferir, envie os detalhes por e-mail diretamente ao mantenedor do projeto com o título `[Vulnerabilidade de Segurança - Nativa]`.

### O que incluir no relatório:
- Descrição clara da vulnerabilidade.
- Passos para reproduzir o problema (código de exemplo `.nat` ou script de teste).
- Possível impacto da falha (ex: estouro de buffer no runtime C99, vazamento de memória mmap, etc.).

### O que esperar:
- **Primeira resposta:** Em até 48 a 72 horas confirmando o recebimento do relato.
- **Avaliação e correção:** Manteremos você informado sobre o andamento e o lançamento da versão corrigida.
