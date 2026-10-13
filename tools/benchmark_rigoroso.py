# tools/benchmark_rigoroso.py
"""
Benchmark Rigoroso de Desempenho, Memória e Fidelidade Numérica:
FP32 Convencional vs. Ternário int8 (Add-only) vs. Ternário Empacotado Base-3 (Vec-LUT 5-trits)

Parâmetros e Condições:
- Model: Qwen2.5-0.5B (Pesos reais da Camada 0)
- Matrizes testadas: gate_proj [4864, 896], down_proj [896, 4864], q_proj [896, 896], k_proj [128, 896]
- Entradas: 10 Embeddings de tokens reais do Qwen2.5 + vetores sintéticos normalizados
- Metodologia de tempo: 20 iterações de aquecimento (warm-up), 100 iterações cronometradas com time.perf_counter_ns
- Estatísticas: Média, Mediana, Desvio Padrão, P25, P75, Mínimo e Máximo
- Perfil de memória: psutil (Process RSS/VMS) + tracemalloc + nbytes + overhead de metadados e LUTs
- Fidelidade numérica: MAE, RMSE, Erro Máximo (L-inf), Erro Relativo L2, Cosine Similarity, Pearson Corr
"""

import os
import gc
import sys
import time
import psutil
import tracemalloc
import numpy as np

# Adicionar pasta raiz ao sys.path para importar nativa
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import nativa

def get_process_memory():
    """Retorna RSS e VMS em bytes do processo atual."""
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    return info.rss, info.vms

def format_bytes(n):
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"

def run_timing_benchmark(fn, warmup=20, runs=100):
    """Executa benchmark com aquecimento e retorna estatísticas em milissegundos."""
    # Warmup
    for _ in range(warmup):
        _ = fn()
    
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        _ = fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000_000.0) # ms
    
    times = np.array(times)
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "std": float(np.std(times)),
        "p25": float(np.percentile(times, 25)),
        "p75": float(np.percentile(times, 75)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }

def compute_numerical_metrics(y_true, y_pred):
    """Calcula métricas rigorosas de fidelidade numérica entre duas saídas."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    
    diff = yt - yp
    abs_diff = np.abs(diff)
    
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(abs_diff))
    
    norm_yt = np.linalg.norm(yt)
    norm_diff = np.linalg.norm(diff)
    rel_l2 = float(norm_diff / (norm_yt + 1e-12))
    
    norm_yp = np.linalg.norm(yp)
    if norm_yt > 1e-12 and norm_yp > 1e-12:
        cosine = float(np.dot(yt, yp) / (norm_yt * norm_yp))
    else:
        cosine = 1.0 if norm_yt == norm_yp else 0.0
        
    std_yt = np.std(yt)
    std_yp = np.std(yp)
    if std_yt > 1e-12 and std_yp > 1e-12:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = 1.0 if std_yt == std_yp else 0.0
        
    return {
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "rel_l2": rel_l2,
        "cosine": cosine,
        "pearson": pearson
    }

def main():
    print("=" * 80)
    print("  BENCHMARK CIENTÍFICO E RIGOROSO: NATIVA v0.3 — KERNELS TERNÁRIOS")
    print("  Comparação: Float32 vs. Ternário int8 (Add-only) vs. Base-3 5-trits (Vec-LUT)")
    print("=" * 80)

    safetensors_path = os.path.join(ROOT_DIR, "data", "qwen2.5_0.5b.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"Erro: Arquivo não encontrado: {safetensors_path}")
        return

    # =========================================================================
    # PARTE 1: MEDIÇÃO REAL DE MEMÓRIA (Heap, Raw Bytes, Objetos e Process RSS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PARTE 1: MEDIÇÃO REAL E TRANSPARENTE DE MEMÓRIA")
    print("=" * 80)

    # Coleta de matrizes para o teste de memória
    shapes_info = [
        ("mlp.gate_proj.weight", "gate_proj"),
        ("mlp.down_proj.weight", "down_proj"),
        ("self_attn.q_proj.weight", "q_proj"),
        ("self_attn.k_proj.weight", "k_proj"),
    ]

    gc.collect()
    rss_base, _ = get_process_memory()

    print(f"[*] RSS Base do Processo Python antes do carregamento: {format_bytes(rss_base)}")
    print("\nDetalhes de Memória por Matriz (Dados brutos + Objeto Tensor + Buffers):")
    print("-" * 80)
    header = f"{'Matriz':<12} | {'Shape':<12} | {'FP32 Raw':<10} | {'T2 int8':<10} | {'5-trits':<10} | {'Obj T2_5t':<10} | {'LUT Buffer':<10} | {'Redução'}"
    print(header)
    print("-" * 80)

    total_fp32_bytes = 0
    total_t2_bytes = 0
    total_p5_bytes = 0
    total_obj_bytes = 0

    cached_matrices = {}

    for tensor_key, name in shapes_info:
        w_fp32 = nativa.load_safetensors_fn(safetensors_path, f"model.layers.0.{tensor_key}")
        shape_str = f"[{w_fp32.shape[0]}, {w_fp32.shape[1]}]"
        
        # Raw bytes
        raw_fp32 = w_fp32.nbytes()
        
        # Quantiza para int8 t2
        w_t2 = w_fp32.quantize_bitnet()
        raw_t2 = w_t2.nbytes()
        
        # Empacota em 5-trits
        w_p5 = w_t2.pack_5trits()
        raw_p5 = w_p5.nbytes()
        
        # Tamanho do objeto Python Tensor (dados + dict + atributos)
        obj_size = sys.getsizeof(w_p5) + sys.getsizeof(w_p5.__dict__) + raw_p5
        
        # Buffer de LUT do kernel Vec-LUT durante a execução:
        # num_groups = (K + 4) // 5. lut01 (9 f32) + lut23 (9 f32) + lut4 (3 f32) = 21 f32 = 84 bytes por grupo
        K = w_fp32.shape[1]
        num_groups = (K + 4) // 5
        lut_buffer_bytes = num_groups * 21 * 4
        
        ratio = raw_fp32 / (raw_p5 + 1e-7)
        
        total_fp32_bytes += raw_fp32
        total_t2_bytes += raw_t2
        total_p5_bytes += raw_p5
        total_obj_bytes += obj_size

        print(f"{name:<12} | {shape_str:<12} | {format_bytes(raw_fp32):<10} | {format_bytes(raw_t2):<10} | {format_bytes(raw_p5):<10} | {format_bytes(obj_size):<10} | {format_bytes(lut_buffer_bytes):<10} | {ratio:.1f}x")
        
        cached_matrices[name] = {
            "fp32": w_fp32,
            "t2": w_t2,
            "p5": w_p5,
            "gamma": w_t2.gamma,
            "shape": w_fp32.shape,
            "lut_buffer_bytes": lut_buffer_bytes
        }

    print("-" * 80)
    print(f"{'SOMA TOTAL':<12} | {'4 Matrizes':<12} | {format_bytes(total_fp32_bytes):<10} | {format_bytes(total_t2_bytes):<10} | {format_bytes(total_p5_bytes):<10} | {format_bytes(total_obj_bytes):<10} | {'Transiente':<10} | {total_fp32_bytes / total_p5_bytes:.1f}x")
    print("-" * 80)

    # Medição real de RSS de uma camada completa
    gc.collect()
    rss_pos_load, _ = get_process_memory()
    print(f"\n[*] RSS do Processo com as matrizes na RAM: {format_bytes(rss_pos_load)} (Delta: +{format_bytes(rss_pos_load - rss_base)})")

    # =========================================================================
    # PARTE 2: BENCHMARK ESTATÍSTICO DE LATÊNCIA EM CONDIÇÕES IDÊNTICAS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PARTE 2: COMPARAÇÃO ESTATÍSTICA DE LATÊNCIA (CONDIÇÕES IDÊNTICAS)")
    print("  Metodologia: 20 iterações de warm-up + 100 iterações cronometradas")
    print("=" * 80)

    # Fixar seed e gerar vetor de ativação de teste para cada dimensão K
    np.random.seed(42)

    for name in ["gate_proj", "down_proj", "q_proj", "k_proj"]:
        mat_info = cached_matrices[name]
        M, K = mat_info["shape"]
        x_vec = np.random.randn(K).astype(np.float32)
        x_tensor = nativa.Tensor(x_vec)

        w_fp32 = mat_info["fp32"]
        w_t2   = mat_info["t2"]
        w_p5   = mat_info["p5"]

        # Callables para benchmark
        fn_fp32 = lambda: w_fp32 @ x_tensor
        fn_t2   = lambda: w_t2 @ x_tensor
        fn_p5   = lambda: w_p5 @ x_tensor

        stats_fp32 = run_timing_benchmark(fn_fp32, warmup=20, runs=100)
        stats_t2   = run_timing_benchmark(fn_t2,   warmup=20, runs=100)
        stats_p5   = run_timing_benchmark(fn_p5,   warmup=20, runs=100)

        speedup_p5_vs_fp32 = stats_fp32["median"] / stats_p5["median"]
        speedup_p5_vs_t2   = stats_t2["median"] / stats_p5["median"]

        print(f"\n--- Matriz: {name} [{M} x {K}] ---")
        print(f"{'Formato':<26} | {'Média (ms)':<10} | {'Mediana':<10} | {'Desv Padrão':<12} | {'IQR [P25..P75]':<15} | {'Min..Max (ms)'}")
        print("-" * 92)
        print(f"{'1. FP32 (OpenBLAS/NumPy)':<26} | {stats_fp32['mean']:<10.3f} | {stats_fp32['median']:<10.3f} | {stats_fp32['std']:<12.3f} | [{stats_fp32['p25']:.3f}..{stats_fp32['p75']:.3f}] ms  | {stats_fp32['min']:.3f}..{stats_fp32['max']:.3f}")
        print(f"{'2. Ternário int8 (Add-only)':<26} | {stats_t2['mean']:<10.3f} | {stats_t2['median']:<10.3f} | {stats_t2['std']:<12.3f} | [{stats_t2['p25']:.3f}..{stats_t2['p75']:.3f}] ms  | {stats_t2['min']:.3f}..{stats_t2['max']:.3f}")
        print(f"{'3. Base-3 5-trits (Vec-LUT)':<26} | {stats_p5['mean']:<10.3f} | {stats_p5['median']:<10.3f} | {stats_p5['std']:<12.3f} | [{stats_p5['p25']:.3f}..{stats_p5['p75']:.3f}] ms  | {stats_p5['min']:.3f}..{stats_p5['max']:.3f}")
        print("-" * 92)
        print(f"    => Speedup do 5-trits (Vec-LUT) vs. FP32:          {speedup_p5_vs_fp32:.2f}x (mediana)")
        print(f"    => Speedup do 5-trits (Vec-LUT) vs. int8 Add-only: {speedup_p5_vs_t2:.2f}x (mediana)")

    # =========================================================================
    # PARTE 3: VALIDAÇÃO RIGOROSA DA FIDELIDADE NUMÉRICA COM MÚLTIPLAS ENTRADAS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PARTE 3: VALIDAÇÃO NUMÉRICA COM ENTRADAS REAIS DO QWEN2.5")
    print("  Testando com 10 Embeddings Reais extraídos do arquivo safetensors")
    print("=" * 80)

    # Carregar tabela de embeddings reais do Qwen2.5
    emb_table = nativa.load_safetensors_fn(safetensors_path, "model.embed_tokens.weight")
    test_token_ids = [100, 500, 1500, 9924, 25000, 50000, 75000, 100000, 125000, 150000]

    norm1 = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.input_layernorm.weight")
    w_gate = cached_matrices["gate_proj"]

    print(f"\nAvaliando projeção mlp.gate_proj ([4864, 896]) para 10 tokens reais pós-RMSNorm:")
    print("-" * 88)
    print(f"{'Token ID':<10} | {'Cosine Sim':<12} | {'Pearson Corr':<12} | {'MAE':<10} | {'RMSE':<10} | {'Max Err':<10} | {'Rel Err L2'}")
    print("-" * 88)

    cosines = []
    pearsons = []
    maes = []
    rmses = []
    max_errs = []
    rel_l2s = []

    for tid in test_token_ids:
        # Embedding real do token
        x_raw = emb_table[tid]
        # Aplica RMSNorm real da camada
        xn = nativa.rmsnorm_fn(x_raw, norm1)
        
        # Executa FP32 vs BitNet 5-trits
        y_fp32 = w_gate["fp32"] @ xn
        y_p5   = w_gate["p5"] @ xn
        
        m = compute_numerical_metrics(y_fp32.data, y_p5.data)
        cosines.append(m["cosine"])
        pearsons.append(m["pearson"])
        maes.append(m["mae"])
        rmses.append(m["rmse"])
        max_errs.append(m["max_err"])
        rel_l2s.append(m["rel_l2"])

        print(f"{tid:<10} | {m['cosine']:<12.5f} | {m['pearson']:<12.5f} | {m['mae']:<10.4f} | {m['rmse']:<10.4f} | {m['max_err']:<10.4f} | {m['rel_l2']*100:.2f}%")

    print("-" * 88)
    print(f"{'MÉDIA':<10} | {np.mean(cosines):<12.5f} | {np.mean(pearsons):<12.5f} | {np.mean(maes):<10.4f} | {np.mean(rmses):<10.4f} | {np.mean(max_errs):<10.4f} | {np.mean(rel_l2s)*100:.2f}%")
    print(f"{'DESVIO':<10} | {np.std(cosines):<12.5f} | {np.std(pearsons):<12.5f} | {np.std(maes):<10.4f} | {np.std(rmses):<10.4f} | {np.std(max_errs):<10.4f} | {np.std(rel_l2s)*100:.2f}%")
    print(f"{'MÍN / MÁX':<10} | {np.min(cosines):.5f}..{np.max(cosines):.5f} | {np.min(pearsons):.5f}..{np.max(pearsons):.5f} | {np.min(maes):.4f}..{np.max(maes):.4f}")

    # =========================================================================
    # PARTE 4: IMPACTO CUMULATIVO NA CAMADA INTEIRA (Atenção + RMSNorm + MLP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PARTE 4: IMPACTO CUMULATIVO NO BLOCO TRANSFORMER COMPLETO")
    print("=" * 80)

    # Carregar camada 0 completa
    w_q_f32 = cached_matrices["q_proj"]["fp32"]
    w_k_f32 = cached_matrices["k_proj"]["fp32"]
    w_v_f32 = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.self_attn.v_proj.weight")
    w_o_f32 = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.self_attn.o_proj.weight")
    w_up_f32 = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.mlp.up_proj.weight")
    w_down_f32 = cached_matrices["down_proj"]["fp32"]
    norm2 = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.post_attention_layernorm.weight")
    
    b_q = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.self_attn.q_proj.bias")
    b_k = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.self_attn.k_proj.bias")
    b_v = nativa.load_safetensors_fn(safetensors_path, "model.layers.0.self_attn.v_proj.bias")

    # Versões empacotadas 5-trits
    w_q_p5 = cached_matrices["q_proj"]["p5"]
    w_k_p5 = cached_matrices["k_proj"]["p5"]
    w_v_p5 = w_v_f32.quantize_bitnet().pack_5trits()
    w_o_p5 = w_o_f32.quantize_bitnet().pack_5trits()
    w_up_p5 = w_up_f32.quantize_bitnet().pack_5trits()
    w_down_p5 = cached_matrices["down_proj"]["p5"]
    w_gate_p5 = cached_matrices["gate_proj"]["p5"]

    layer_cosines = []
    layer_pearsons = []
    layer_rel_errs = []

    for tid in test_token_ids:
        x_in = emb_table[tid]
        
        # 1. Forward Pass FP32
        xn1 = nativa.rmsnorm_fn(x_in, norm1)
        q_f = (w_q_f32 @ xn1) + b_q
        k_f = (w_k_f32 @ xn1) + b_k
        v_f = (w_v_f32 @ xn1) + b_v
        v0_f = v_f.data[0:64]
        v1_f = v_f.data[64:128]
        v_rep_f = nativa.Tensor(np.concatenate([v0_f]*7 + [v1_f]*7))
        attn_out_f = w_o_f32 @ v_rep_f
        x_mid_f = x_in + attn_out_f
        
        xn2_f = nativa.rmsnorm_fn(x_mid_f, norm2)
        gate_f = w_gate["fp32"] @ xn2_f
        up_f = w_up_f32 @ xn2_f
        h_f = nativa.silu_fn(gate_f) * up_f
        mlp_out_f = w_down_f32 @ h_f
        y_f32_layer = x_mid_f + mlp_out_f
        
        # 2. Forward Pass BitNet 5-trits
        xn1_b = nativa.rmsnorm_fn(x_in, norm1)
        q_b = (w_q_p5 @ xn1_b) + b_q
        k_b = (w_k_p5 @ xn1_b) + b_k
        v_b = (w_v_p5 @ xn1_b) + b_v
        v0_b = v_b.data[0:64]
        v1_b = v_b.data[64:128]
        v_rep_b = nativa.Tensor(np.concatenate([v0_b]*7 + [v1_b]*7))
        attn_out_b = w_o_p5 @ v_rep_b
        x_mid_b = x_in + attn_out_b
        
        xn2_b = nativa.rmsnorm_fn(x_mid_b, norm2)
        gate_b = w_gate_p5 @ xn2_b
        up_b = w_up_p5 @ xn2_b
        h_b = nativa.silu_fn(gate_b) * up_b
        mlp_out_b = w_down_p5 @ h_b
        y_bit_layer = x_mid_b + mlp_out_b
        
        m_layer = compute_numerical_metrics(y_f32_layer.data, y_bit_layer.data)
        layer_cosines.append(m_layer["cosine"])
        layer_pearsons.append(m_layer["pearson"])
        layer_rel_errs.append(m_layer["rel_l2"])

    print(f"Fidelidade na saída final do Bloco Transformer (Camada 0) para os 10 tokens:")
    print(f"  - Similaridade de Cosseno Média: {np.mean(layer_cosines):.4f} (Mín: {np.min(layer_cosines):.4f}, Máx: {np.max(layer_cosines):.4f})")
    print(f"  - Correlação de Pearson Média:   {np.mean(layer_pearsons):.4f} (Mín: {np.min(layer_pearsons):.4f}, Máx: {np.max(layer_pearsons):.4f})")
    print(f"  - Erro Relativo L2 Médio:        {np.mean(layer_rel_errs)*100:.2f}%")

    print("\n" + "=" * 80)
    print("  RELATÓRIO DE CONCLUSÃO DO BENCHMARK RIGOROSO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

if __name__ == "__main__":
    main()
