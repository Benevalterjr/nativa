#!/usr/bin/env python3
"""
qutrit_test/test_branch_dev_decisions.py
Evaluates the real-world decision capabilities of the INT4-quantized Qwen model (data/qwen2.5_1.5b.natq4)
against the 10 decision and risk test cases from the Branch.dev framework (https://github.com/Benevalterjr/branch.dev).

Measures:
  1. Decision Accuracy (Winner choice matches ground truth expected)
  2. Latency (ms) and Throughput (tokens/s) on CPU
  3. Structured Choice Reliability
"""

import os
import re
import sys
import json
import time
import tempfile
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nativa
from tools.chat_natq4 import write_vocab_file

MODEL_PATH = os.path.join(REPO_ROOT, "data", "qwen2.5_1.5b.natq4")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "data", "qwen2.5_1.5b.natq4.tokenizer.json")
ENGINE_PATH = os.path.join(REPO_ROOT, "tools", "nativa_engine.exe")

# 10 Real Cases from branch.dev (Trading B3, Financial Risk, SRE, Fraud, Churn, Support)
TEST_CASES = [
    {
        "id": "TRD-01",
        "category": "Trading & B3",
        "name": "PETR4 — Consolidação em Resistência",
        "state": {
            "ticker": "PETR4.SA",
            "preco": 48.42,
            "rsi14": 58.62,
            "sma9": 48.85,
            "sma21": 46.77,
            "cenario": "preço abaixo da resistência SMA9 em consolidação sem volume forte de rompimento"
        },
        "choices": {
            "manter_neutro": "consolidação técnica entre médias sem volume de rompimento, prudência em aguardar",
            "comprar_agressivo": "rompimento confirmado com volume institucional massivo acima das resistências",
            "vender_despejo": "perda de suporte crítico da SMA21 com fluxo vendedor pesado"
        },
        "task": "determinar decisão de trading para a posição em PETR4",
        "expected": "manter_neutro"
    },
    {
        "id": "TRD-02",
        "category": "Trading & B3",
        "name": "VALE3 — Detecção de Bull Trap",
        "state": {
            "ticker": "VALE3.SA",
            "preco": 62.10,
            "evento": "preço furou topo anterior por 3 centavos",
            "volumeDelta": -0.45,
            "divergenciaRSI": "baixa clara no tempo gráfico de 15min",
            "saldoAgressao": -4800000
        },
        "choices": {
            "alerta_bull_trap": "falso rompimento de topo sem confirmação de volume e com agressão vendedora",
            "compra_rompimento": "rompimento legítimo de topo histórico com fluxo comprador acelerado",
            "mercado_lateral": "mercado estável sem sinais operacionais relevantes"
        },
        "task": "classificar o padrão de rompimento técnico",
        "expected": "alerta_bull_trap"
    },
    {
        "id": "RISK-01",
        "category": "Risco Financeiro",
        "name": "Gestão de Risco — Drawdown Diário Limite",
        "state": {
            "carteira": "Fundo Quantitativo L/S",
            "drawdownDiario": "-3.95%",
            "limiteDiarioRisco": "-4.00%",
            "eventoNoticioso": "Abertura de investigação regulatória com leilão de volatilidade",
            "spreadLivro": "abertura atípica de 2.4%"
        },
        "choices": {
            "estancar_risco": "interrupção preventiva imediata e fechamento de risco no limite de perda diário",
            "dobrar_aposta": "aumento de margem para abaixar preço médio acreditando em exagero",
            "ignorar_noticia": "manter posições abertas sem nenhuma intervenção"
        },
        "task": "executar protocolo de controle de risco de capital",
        "expected": "estancar_risco"
    },
    {
        "id": "FIN-01",
        "category": "Risco Financeiro",
        "name": "Conciliação — Cobrança Duplicada Gateway",
        "state": {
            "clienteId": "USR-9921",
            "valor": 249.90,
            "ocorrencia": "duas cobranças idênticas em intervalo de 4 segundos",
            "gatewayStatus": "ambas capturadas",
            "historicoCliente": "18 meses adimplente, zero chargebacks"
        },
        "choices": {
            "estorno_automatico": "devolução imediata da segunda transação duplicada de forma automatizada",
            "bloqueio_fraude": "cancelar conta do cliente por suspeita de fraude",
            "recusar_contato": "negar devolução alegando que cobrança foi processada"
        },
        "task": "definir resolução do incidente financeiro",
        "expected": "estorno_automatico"
    },
    {
        "id": "INFRA-01",
        "category": "Infra & SRE",
        "name": "SRE — Saturação de Conexões em Produção",
        "state": {
            "servico": "Checkout & Pagamentos",
            "postgresPoolAtivo": "100%",
            "pedidosFilaEspera": 1840,
            "latenciaP99": "8400ms",
            "erro500Taxa": "14.2%"
        },
        "choices": {
            "acionamento_sre_critico": "incidente P0 de produção com bloqueio de receita, acionar plantão e failover",
            "observacao_passiva": "oscilação passageira de tráfego, registrar apenas em log informativo",
            "agendar_reuniao": "marcar alinhamento na semana que vem para avaliar capacidade"
        },
        "task": "classificar severidade e resposta para o alerta de infraestrutura",
        "expected": "acionamento_sre_critico"
    },
    {
        "id": "SEC-01",
        "category": "Segurança",
        "name": "Defesa — Ataque Card Testing via Tor",
        "state": {
            "ipOrigem": "Tor Exit Node",
            "tentativasMinuto": 48,
            "cartoesDistintos": 15,
            "taxaRejeicao": "94%",
            "valorTransacao": "R$ 1,50"
        },
        "choices": {
            "bloqueio_imediato_waf": "ataque automatizado de teste de cartões furtados, bloquear IP e fingerprint no WAF",
            "desafio_captcha": "exibir captcha simples para o bot",
            "permitir_transacoes": "aprovar pagamentos normalmente"
        },
        "task": "determinar medida de contenção de segurança",
        "expected": "bloqueio_imediato_waf"
    },
    {
        "id": "CASE-01",
        "category": "Suporte / Financeiro",
        "name": "Estorno de Cobrança Duplicada",
        "state": {
            "assunto": "Fui cobrado duas vezes no cartão Master",
            "mensagem": "Olá, minha assinatura deste mês veio duas vezes na mesma data. Exijo estorno da segunda cobrança de R$ 99 imediatamente.",
            "valorDuplicado": 99.00,
            "diasAtraso": 0
        },
        "choices": {
            "aprovar_estorno": "solicitação clara de estorno ou reembolso de cobrança indevida ou duplicada",
            "suporte_tecnico": "problemas técnicos, erros no sistema, telas travadas ou falhas",
            "duvidas_comerciais": "informações de preços, novos planos ou contratação"
        },
        "task": "determinar ação para o ticket financeiro",
        "expected": "aprovar_estorno"
    },
    {
        "id": "CASE-02",
        "category": "Customer Success",
        "name": "Detecção de Risco de Churn",
        "state": {
            "usuario": "TechCorp",
            "diasSemAcessar": 45,
            "ticketsAbertosSemResposta: 3": 3,
            "quedaDeUsoPercentual": 80,
            "plano": "Enterprise"
        },
        "choices": {
            "alto_risco": "cliente inativo há muito tempo, com chamados parados e queda drástica de uso",
            "medio_risco": "cliente com atividade reduzida mas estável",
            "baixo_risco": "cliente saudável e satisfeito"
        },
        "task": "avaliar probabilidade de cancelamento do cliente",
        "expected": "alto_risco"
    },
    {
        "id": "CASE-03",
        "category": "Infraestrutura",
        "name": "Alerta Crítico de Banco de Dados",
        "state": {
            "servico": "Auth Service",
            "cluster": "prod-us-east-1",
            "erro": "Conexões com Postgres esgotadas, pool zerado, latência p99 > 8000ms",
            "impacto": "Usuários não conseguem fazer login"
        },
        "choices": {
            "incidente_critico": "falha grave em produção, indisponibilidade de serviço ou banco travado",
            "alerta_moderado": "lentidão isolada ou aviso preventivo",
            "informativo": "manutenção rotineira ou log informativo"
        },
        "task": "classificar severidade do alerta operacional",
        "expected": "incidente_critico"
    },
    {
        "id": "CASE-04",
        "category": "Segurança / Antifraude",
        "name": "Tentativa de Fraude / Abuso de Cartão",
        "state": {
            "ip": "185.220.101.5",
            "pais": "Desconhecido (VPN)",
            "tentativasCartao10Min": 14,
            "cartoesDiferentesUsados": 8,
            "cvvInvalidos": 6
        },
        "choices": {
            "bloqueio_fraude": "ataque de card testing, múltiplas tentativas com cartões diferentes e VPN",
            "reanalise_manual": "transação suspeita necessitando documento",
            "aprovacao_normal": "compra legítima de cliente confiável"
        },
        "task": "avaliar risco de fraude da transação",
        "expected": "bloqueio_fraude"
    }
]

def norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())

def run_branch_dev_benchmark():
    print("=" * 115)
    print("  AVALIAÇÃO DE CAPACIDADE DO MODELO INT4 (QWEN2.5-1.5B) COM CASOS REAIS DO BRANCH.DEV")
    print("=" * 115)

    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Modelo INT4 não encontrado em {MODEL_PATH}")
        return
    if not os.path.isfile(TOKENIZER_PATH):
        print(f"[ERROR] Tokenizer não encontrado em {TOKENIZER_PATH}")
        return

    print(f"[*] Carregando tokenizer de {TOKENIZER_PATH}...")
    tokenizer = nativa.Tokenizer(TOKENIZER_PATH)

    vocab_tmp = tempfile.NamedTemporaryFile(prefix="nativa-vocab-", suffix=".bin", delete=False)
    vocab_path = vocab_tmp.name
    vocab_tmp.close()

    try:
        write_vocab_file(tokenizer, vocab_path)
    except Exception as e:
        print(f"[ERROR] Falha ao exportar vocabulário: {e}")
        return

    # Start persistent native engine process
    cmd_engine = [
        os.path.abspath(ENGINE_PATH),
        "--",
        os.path.abspath(MODEL_PATH),
        "--chat",
        "--vocab",
        os.path.abspath(vocab_path)
    ]
    env = os.environ.copy()
    env["OMP_WAIT_POLICY"] = "PASSIVE"

    print(f"[*] Inicializando motor nativo C99...")
    proc = subprocess.Popen(
        cmd_engine,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=env
    )

    # Wait for ready signal
    while True:
        line = proc.stdout.readline()
        if not line:
            print("[ERROR] Falha ao iniciar motor nativo.")
            return
        if "[NATIVA_CHAT_READY]" in line:
            break

    print("[+] Motor Nativa INT4 pronto para inferência!\n")

    results = []
    correct_count = 0
    total_time_gen = 0.0
    total_tokens_gen = 0

    print("-" * 115)
    print(f"{'Caso ID':<8} | {'Categoria':<18} | {'Esperado':<24} | {'Previsto':<24} | {'Status':<7} | {'Latência':<9} | {'Throughput':<9}")
    print("-" * 115)

    for c in TEST_CASES:
        # Reset KV cache context before each independent decision case
        proc.stdin.write("RESET\n")
        proc.stdin.flush()

        # Build prompt identical to branch.dev benchmark
        options_text = "\n".join([f'- "{k}": {desc}' for k, desc in c["choices"].items()])
        prompt_user = (
            f"Você é um motor analítico determinístico de alta precisão.\n"
            f"Estado atual:\n{json.dumps(c['state'], ensure_ascii=False, indent=2)}\n\n"
            f"Tarefa de Decisão: {c['task']}\n\n"
            f"Opções Disponíveis:\n{options_text}\n\n"
            f"Instrução: Escolha a melhor opção técnica. Responda ESTRITAMENTE com o identificador da opção (apenas o nome exato entre as opções, sem comentários)."
        )

        full_prompt = (
            "<|im_start|>system\n"
            "Você é um motor analítico determinístico de alta precisão. Responda estritamente com o identificador exato da opção escolhida.<|im_end|>\n"
            f"<|im_start|>user\n{prompt_user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        tokens = tokenizer.encode(full_prompt).tolist()
        max_gen = 15
        cmd = f"INPUT {len(tokens)} {max_gen} " + " ".join(map(str, tokens)) + "\n"

        t0 = time.perf_counter()
        proc.stdin.write(cmd)
        proc.stdin.flush()

        # Read until TURN_DONE
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            buf += ch
            if "[TURN_DONE" in buf and "]" in buf[buf.find("[TURN_DONE"): ]:
                break

        dt = (time.perf_counter() - t0) * 1000.0

        # Parse speed and tokens from TURN_DONE
        m = re.search(r"tokens=(\d+)\s+time=([0-9.]+)\s+speed=([0-9.]+)", buf)
        tok_gen = int(m.group(1)) if m else 0
        speed_tok_s = float(m.group(3)) if m else 0.0

        total_time_gen += dt
        total_tokens_gen += tok_gen

        # Extract raw answer before TURN_DONE
        raw_answer = buf[:buf.find("[TURN_DONE")].strip()
        raw_norm = norm(raw_answer)

        # Match winner choice
        winner = "indefinido"
        # 1. Exact substring
        for k in c["choices"].keys():
            if k.lower() in raw_answer.lower():
                winner = k
                break
        # 2. Normalized match (handles underscore drops or spacing)
        if winner == "indefinido":
            for k in c["choices"].keys():
                if norm(k) in raw_norm:
                    winner = k
                    break

        is_correct = (winner == c["expected"])
        if is_correct:
            correct_count += 1

        status_str = " ACERTO " if is_correct else " ERRO "
        print(f"{c['id']:<8} | {c['category']:<18} | {c['expected']:<24} | {str(winner):<24} | {status_str:<7} | {dt:>6.0f} ms | {speed_tok_s:>5.1f} tok/s")

        results.append({
            "id": c["id"],
            "name": c["name"],
            "category": c["category"],
            "expected": c["expected"],
            "winner": winner,
            "is_correct": is_correct,
            "latency_ms": round(dt, 1),
            "tokens": tok_gen,
            "speed": speed_tok_s,
            "raw": raw_answer
        })

    # Close engine
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=2)
    except Exception:
        proc.kill()

    try:
        os.unlink(vocab_path)
    except OSError:
        pass

    print("-" * 115)
    acc = (correct_count / len(TEST_CASES)) * 100.0
    avg_lat = total_time_gen / len(TEST_CASES)
    avg_speed = total_tokens_gen / (total_time_gen / 1000.0) if total_time_gen > 0 else 0.0

    print("\n" + "=" * 60)
    print("  RESULTADO FINAL DO BENCHMARK BRANCH.DEV")
    print("=" * 60)
    print(f"  Total de Casos Avaliados : {len(TEST_CASES)}")
    print(f"  Decisões Corretas        : {correct_count} / {len(TEST_CASES)}")
    print(f"  Acurácia de Decisão      : {acc:.1f}%")
    print(f"  Latência Média por Caso  : {avg_lat:.1f} ms")
    print(f"  Throughput de Inferência : {avg_speed:.1f} tokens/s (CPU)")
    print("=" * 60)

    # Save benchmark results to JSON
    out_file = os.path.join(REPO_ROOT, "qutrit_test", "branch_dev_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen2.5-1.5B-Instruct-INT4 (NATQ4)",
            "accuracy_percent": acc,
            "correct_count": correct_count,
            "total_cases": len(TEST_CASES),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_speed_tok_s": round(avg_speed, 1),
            "cases": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Relatório detalhado salvo em: {out_file}\n")

if __name__ == "__main__":
    run_branch_dev_benchmark()
