/* ========================================================================= */
/* Gerado automaticamente pelo compilador Nativa (AOT C99 Transpiler)        */
/* ========================================================================= */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#ifndef NAT_PI
#define NAT_PI 3.14159265358979323846
#endif
#ifndef NAT_E
#define NAT_E  2.71828182845904523536
#endif
/* Primitivas de quantização ternária e empacotamento {-1, 0, +1} em C */
static inline int8_t* nat_quantize_ternary(const double* v, int64_t length, double threshold) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        if (v[i] > threshold) out[i] = 1;
        else if (v[i] < -threshold) out[i] = -1;
        else out[i] = 0;
    }
    return out;
}
static inline uint8_t* nat_pack_ternary(const int8_t* in, int64_t length) {
    int64_t num_bytes = (length + 3) / 4;
    uint8_t* out = (uint8_t*)calloc(num_bytes, sizeof(uint8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        uint8_t enc = (in[i] == 1) ? 1 : ((in[i] == -1) ? 2 : 0);
        int64_t b = i / 4;
        int shift = (i % 4) * 2;
        out[b] |= (enc << shift);
    }
    return out;
}
static inline int8_t* nat_unpack_ternary(const uint8_t* in, int64_t length) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    static const int8_t map[4] = {0, 1, -1, 0};
    for (int64_t i = 0; i < length; i++) {
        int64_t b = i / 4;
        int shift = (i % 4) * 2;
        uint8_t code = (in[b] >> shift) & 0x03;
        out[i] = map[code];
    }
    return out;
}
/* Funções matemáticas e primitivas de IA */
static inline double nat_ln(double x) { return log(x); }
static inline double nat_relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double nat_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double nat_silu(double x) { return x / (1.0 + exp(-x)); }

/* Kernel Add-Only Matmul Ternário {-1, 0, +1} em C */
static inline void nat_ternary_matmul_vec(const int8_t* W, const double* x, double* y, int64_t M, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const int8_t* row = W + i * K;
        for (int64_t k = 0; k < K; k++) {
            int8_t w = row[k];
            if (w == 1) acc += x[k];
            else if (w == -1) acc -= x[k];
        }
        y[i] = acc;
    }
}

/* BitNet b1.58: Empacotamento base-3 denso (5 trits/byte = 1.60 bits/trit) */
static inline void nat_pack_5trits(const int8_t* in, uint8_t* out_bytes, int64_t length) {
    int64_t out_idx = 0;
    for (int64_t i = 0; i < length; i += 5) {
        uint16_t b_val = 0;
        uint16_t mul = 1;
        for (int j = 0; j < 5; j++) {
            int8_t t = (i + j < length) ? (in[i + j] + 1) : 1;
            b_val += (uint16_t)(t * mul);
            mul *= 3;
        }
        out_bytes[out_idx++] = (uint8_t)b_val;
    }
}

static inline void nat_unpack_5trits(const uint8_t* in_bytes, int8_t* out, int64_t length) {
    int64_t out_idx = 0;
    int64_t num_bytes = (length + 4) / 5;
    for (int64_t b = 0; b < num_bytes && out_idx < length; b++) {
        uint8_t val = in_bytes[b];
        for (int step = 0; step < 5 && out_idx < length; step++) {
            out[out_idx++] = (int8_t)(val % 3) - 1;
            val /= 3;
        }
    }
}

/* Fused Ternary GEMV: Decodifica 5 trits por byte em tempo real em registradores, streaming sem buffer uncompressed */
static inline void nat_fused_gemv_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {
    int64_t bytes_per_row = (K + 4) / 5;
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const uint8_t* row_packed = W_packed + i * bytes_per_row;
        int64_t k = 0;
        for (int64_t b = 0; b < bytes_per_row && k < K; b++) {
            uint8_t val = row_packed[b];
            for (int step = 0; step < 5 && k < K; step++) {
                int8_t trit = (int8_t)(val % 3) - 1;
                val /= 3;
                if (trit == 1) acc += x[k];
                else if (trit == -1) acc -= x[k];
                k++;
            }
        }
        y[i] = acc * gamma;
    }
}

/* Vec-LUT Ternary GEMV: Look-Up Table por pares de trits com aceleração de streaming em C */
static inline void nat_gemv_vec_lut_5trits(const uint8_t* W_packed, const double* x, double* y, int64_t M, int64_t K, double gamma) {
    int64_t num_groups = (K + 4) / 5;
    if (num_groups <= 0) return;
    double* lut = (double*)malloc(num_groups * 21 * sizeof(double));
    if (!lut) {
        return;
    }
    double* lut01 = lut;
    double* lut23 = lut + num_groups * 9;
    double* lut4  = lut + num_groups * 18;
    for (int64_t g = 0; g < num_groups; g++) {
        int64_t base = g * 5;
        double x0 = (base < K) ? x[base] : 0.0;
        double x1 = (base + 1 < K) ? x[base + 1] : 0.0;
        double x2 = (base + 2 < K) ? x[base + 2] : 0.0;
        double x3 = (base + 3 < K) ? x[base + 3] : 0.0;
        double x4 = (base + 4 < K) ? x[base + 4] : 0.0;
        for (int t1 = 0; t1 < 3; t1++) {
            for (int t0 = 0; t0 < 3; t0++) {
                lut01[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x0 + (t1 - 1) * x1;
                lut23[g * 9 + (t0 + 3 * t1)] = (t0 - 1) * x2 + (t1 - 1) * x3;
            }
        }
        for (int t4 = 0; t4 < 3; t4++) {
            lut4[g * 3 + t4] = (t4 - 1) * x4;
        }
    }
    for (int64_t i = 0; i < M; i++) {
        double acc = 0.0;
        const uint8_t* row = W_packed + i * num_groups;
        for (int64_t g = 0; g < num_groups; g++) {
            uint8_t b = row[g];
            uint8_t idx01 = b % 9;
            uint8_t idx23 = (b / 9) % 9;
            uint8_t idx4  = b / 81;
            acc += lut01[g * 9 + idx01] + lut23[g * 9 + idx23] + lut4[g * 3 + idx4];
        }
        y[i] = acc * gamma;
    }
    free(lut);
}

/* Wrappers de Alto Nível de BitNet b1.58 para o Transpilador C */
static inline uint8_t* nat_pack_ternary_5trits(const int8_t* in, int64_t length) {
    int64_t num_bytes = (length + 4) / 5;
    uint8_t* out = (uint8_t*)malloc(num_bytes * sizeof(uint8_t));
    if (out) nat_pack_5trits(in, out, length);
    return out;
}
static inline int8_t* nat_unpack_ternary_5trits(const uint8_t* in, int64_t length) {
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (out) nat_unpack_5trits(in, out, length);
    return out;
}
static inline double* nat_gemv_vec_lut(const uint8_t* W_packed, const double* x, int64_t M, int64_t K) {
    double* y = (double*)malloc(M * sizeof(double));
    if (y) nat_gemv_vec_lut_5trits(W_packed, x, y, M, K, 1.0);
    return y;
}
static inline double nat_tensor_absmean(const double* W, int64_t length) {
    if (length <= 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < length; i++) s += fabs(W[i]);
    return s / (double)length;
}
static inline int8_t* nat_quantize_bitnet(const double* W, int64_t length) {
    double gamma = nat_tensor_absmean(W, length);
    double inv = 1.0 / (gamma + 1e-7);
    int8_t* out = (int8_t*)malloc(length * sizeof(int8_t));
    if (!out) return NULL;
    for (int64_t i = 0; i < length; i++) {
        double sc = W[i] * inv;
        double r = round(sc);
        if (r > 1.0) r = 1.0; else if (r < -1.0) r = -1.0;
        out[i] = (int8_t)r;
    }
    return out;
}
static inline double nat_mae(const double* a, const double* b, int64_t length) {
    if (length <= 0) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < length; i++) s += fabs(a[i] - b[i]);
    return s / (double)length;
}
static inline double nat_relative_error(const double* a, const double* b, int64_t length) {
    double norm_diff = 0.0, norm_a = 0.0;
    for (int64_t i = 0; i < length; i++) {
        double d = a[i] - b[i];
        norm_diff += d * d; norm_a += a[i] * a[i];
    }
    return norm_a > 0.0 ? sqrt(norm_diff / norm_a) : 0.0;
}
static inline double nat_pearson_corr(const double* a, const double* b, int64_t length) {
    if (length < 2) return 1.0;
    double ma = 0.0, mb = 0.0;
    for (int64_t i = 0; i < length; i++) { ma += a[i]; mb += b[i]; }
    ma /= (double)length; mb /= (double)length;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int64_t i = 0; i < length; i++) {
        double diff_a = a[i] - ma; double diff_b = b[i] - mb;
        num += diff_a * diff_b; da += diff_a * diff_a; db += diff_b * diff_b;
    }
    double den = sqrt(da * db);
    return den > 0.0 ? (num / den) : 0.0;
}
static inline double nat_cosine_sim(const double* a, const double* b, int64_t length) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < length; i++) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    double den = sqrt(na) * sqrt(nb);
    return den > 0.0 ? (dot / den) : 0.0;
}


double theta(double t);
double hardy_z_aprox(double t);
double bissecao(double a_in, double b_in);

double theta(double t) {
    return ((((((((((((0.5) * (t))) * (nat_ln(((t) / (((2.0) * (NAT_PI)))))))) - (((0.5) * (t))))) - (((NAT_PI) / (8.0))))) + (((1.0) / (((48.0) * (t))))))) + (((7.0) / (((((((5760.0) * (t))) * (t))) * (t))))));
}

double hardy_z_aprox(double t) {
    int64_t N = ((int64_t)(floor(sqrt(((t) / (((2.0) * (NAT_PI))))))));
    double fase = theta(t);
    double soma = 0.0;
    for (int64_t n = 1LL; n < ((N) + (1LL)); n++) {
        double nf = ((double)(n));
        soma += ((cos(((fase) - (((t) * (nat_ln(nf))))))) / (sqrt(nf)));
    }
    return ((2.0) * (soma));
}

double bissecao(double a_in, double b_in) {
    double a = a_in;
    double b = b_in;
    double fa = hardy_z_aprox(a);
    for (int64_t passo = 0LL; passo < 60LL; passo++) {
        double meio = ((((a) + (b))) / (2.0));
        double fm = hardy_z_aprox(meio);
        if (((fm) == (0.0))) {
            return meio;
        }
        if (((((fa) * (fm))) < (0.0))) {
            b = meio;
        } else {
            a = meio;
            fa = fm;
        }
    }
    return ((((a) + (b))) / (2.0));
}

int main(int argc, char** argv) {
    double inicio = 10.0;
    double fim = 100.0;
    double passo = 0.25;
    printf("=== Buscando Zeros da Função Zeta na Linha Critica [t = %f ate %f] ===\n", (double)(inicio), (double)(fim));
    double anterior_t = inicio;
    double anterior_z = hardy_z_aprox(anterior_t);
    int64_t encontrados = 0LL;
    double t = ((inicio) + (passo));
    while (((t) <= (fim))) {
        double z = hardy_z_aprox(t);
        if (((((anterior_z) * (z))) < (0.0))) {
            encontrados += 1LL;
            double raiz = bissecao(anterior_t, t);
            printf("[%2lld] Zero detectado: t = %.6f\n", (long long)(encontrados), (double)(raiz));
        }
        anterior_t = t;
        anterior_z = z;
        t += passo;
    }
    printf("=== Busca concluida: %lld zeros nao-triviais encontrados ===\n", (long long)(encontrados));
    return 0;
}
