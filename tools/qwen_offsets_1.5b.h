#ifndef QWEN_OFFSETS_1_5B_H
#define QWEN_OFFSETS_1_5B_H

#include <stdint.h>

#define QWEN15_VOCAB_SIZE 151936
#define QWEN15_DIM 1536
#define QWEN15_INTERMEDIATE_DIM 8960
#define QWEN15_NUM_LAYERS 28
#define QWEN15_NUM_HEADS 12
#define QWEN15_NUM_KV_HEADS 2
#define QWEN15_HEAD_DIM 128
#define QWEN15_ROPE_THETA 1000000.0f
#define QWEN15_RMS_EPS 1e-6f

typedef struct {
    uint64_t norm1;
    uint64_t norm2;
    uint64_t wq;
    uint64_t bq;
    uint64_t wk;
    uint64_t bk;
    uint64_t wv;
    uint64_t bv;
    uint64_t wo;
    uint64_t wgate;
    uint64_t wup;
    uint64_t wdown;
} Qwen15LayerOffsets;

typedef struct {
    uint64_t embed_tokens;
    uint64_t norm;
    Qwen15LayerOffsets layers[QWEN15_NUM_LAYERS];
} Qwen15ModelOffsets;

static const Qwen15ModelOffsets g_qwen15_offsets = {
    .embed_tokens = 38536ULL,
    .norm = 3087464072ULL,
    .layers = {
        // Camada 0
        {
            .norm1 = 466785928ULL,
            .norm2 = 549364360ULL,
            .wq = 554876040ULL,
            .bq = 554872968ULL,
            .wk = 549367944ULL,
            .bk = 549367432ULL,
            .wv = 559595144ULL,
            .bv = 559594632ULL,
            .wo = 550154376ULL,
            .wgate = 494314120ULL,
            .wup = 521839240ULL,
            .wdown = 466789000ULL,
        },
        // Camada 1
        {
            .norm1 = 560381576ULL,
            .norm2 = 642960008ULL,
            .wq = 648471688ULL,
            .bq = 648468616ULL,
            .wk = 642963592ULL,
            .bk = 642963080ULL,
            .wv = 653190792ULL,
            .bv = 653190280ULL,
            .wo = 643750024ULL,
            .wgate = 587909768ULL,
            .wup = 615434888ULL,
            .wdown = 560384648ULL,
        },
        // Camada 2
        {
            .norm1 = 1589933704ULL,
            .norm2 = 1672512136ULL,
            .wq = 1678023816ULL,
            .bq = 1678020744ULL,
            .wk = 1672515720ULL,
            .bk = 1672515208ULL,
            .wv = 1682742920ULL,
            .bv = 1682742408ULL,
            .wo = 1673302152ULL,
            .wgate = 1617461896ULL,
            .wup = 1644987016ULL,
            .wdown = 1589936776ULL,
        },
        // Camada 3
        {
            .norm1 = 2432294536ULL,
            .norm2 = 2514872968ULL,
            .wq = 2520384648ULL,
            .bq = 2520381576ULL,
            .wk = 2514876552ULL,
            .bk = 2514876040ULL,
            .wv = 2525103752ULL,
            .bv = 2525103240ULL,
            .wo = 2515662984ULL,
            .wgate = 2459822728ULL,
            .wup = 2487347848ULL,
            .wdown = 2432297608ULL,
        },
        // Camada 4
        {
            .norm1 = 2525890184ULL,
            .norm2 = 2608468616ULL,
            .wq = 2613980296ULL,
            .bq = 2613977224ULL,
            .wk = 2608472200ULL,
            .bk = 2608471688ULL,
            .wv = 2618699400ULL,
            .bv = 2618698888ULL,
            .wo = 2609258632ULL,
            .wgate = 2553418376ULL,
            .wup = 2580943496ULL,
            .wdown = 2525893256ULL,
        },
        // Camada 5
        {
            .norm1 = 2619485832ULL,
            .norm2 = 2702064264ULL,
            .wq = 2707575944ULL,
            .bq = 2707572872ULL,
            .wk = 2702067848ULL,
            .bk = 2702067336ULL,
            .wv = 2712295048ULL,
            .bv = 2712294536ULL,
            .wo = 2702854280ULL,
            .wgate = 2647014024ULL,
            .wup = 2674539144ULL,
            .wdown = 2619488904ULL,
        },
        // Camada 6
        {
            .norm1 = 2713081480ULL,
            .norm2 = 2795659912ULL,
            .wq = 2801171592ULL,
            .bq = 2801168520ULL,
            .wk = 2795663496ULL,
            .bk = 2795662984ULL,
            .wv = 2805890696ULL,
            .bv = 2805890184ULL,
            .wo = 2796449928ULL,
            .wgate = 2740609672ULL,
            .wup = 2768134792ULL,
            .wdown = 2713084552ULL,
        },
        // Camada 7
        {
            .norm1 = 2806677128ULL,
            .norm2 = 2889255560ULL,
            .wq = 2894767240ULL,
            .bq = 2894764168ULL,
            .wk = 2889259144ULL,
            .bk = 2889258632ULL,
            .wv = 2899486344ULL,
            .bv = 2899485832ULL,
            .wo = 2890045576ULL,
            .wgate = 2834205320ULL,
            .wup = 2861730440ULL,
            .wdown = 2806680200ULL,
        },
        // Camada 8
        {
            .norm1 = 2900272776ULL,
            .norm2 = 2982851208ULL,
            .wq = 2988362888ULL,
            .bq = 2988359816ULL,
            .wk = 2982854792ULL,
            .bk = 2982854280ULL,
            .wv = 2993081992ULL,
            .bv = 2993081480ULL,
            .wo = 2983641224ULL,
            .wgate = 2927800968ULL,
            .wup = 2955326088ULL,
            .wdown = 2900275848ULL,
        },
        // Camada 9
        {
            .norm1 = 2993868424ULL,
            .norm2 = 3076446856ULL,
            .wq = 3081958536ULL,
            .bq = 3081955464ULL,
            .wk = 3076450440ULL,
            .bk = 3076449928ULL,
            .wv = 3086677640ULL,
            .bv = 3086677128ULL,
            .wo = 3077236872ULL,
            .wgate = 3021396616ULL,
            .wup = 3048921736ULL,
            .wdown = 2993871496ULL,
        },
        // Camada 10
        {
            .norm1 = 653977224ULL,
            .norm2 = 736555656ULL,
            .wq = 742067336ULL,
            .bq = 742064264ULL,
            .wk = 736559240ULL,
            .bk = 736558728ULL,
            .wv = 746786440ULL,
            .bv = 746785928ULL,
            .wo = 737345672ULL,
            .wgate = 681505416ULL,
            .wup = 709030536ULL,
            .wdown = 653980296ULL,
        },
        // Camada 11
        {
            .norm1 = 747572872ULL,
            .norm2 = 830151304ULL,
            .wq = 835662984ULL,
            .bq = 835659912ULL,
            .wk = 830154888ULL,
            .bk = 830154376ULL,
            .wv = 840382088ULL,
            .bv = 840381576ULL,
            .wo = 830941320ULL,
            .wgate = 775101064ULL,
            .wup = 802626184ULL,
            .wdown = 747575944ULL,
        },
        // Camada 12
        {
            .norm1 = 841168520ULL,
            .norm2 = 923746952ULL,
            .wq = 929258632ULL,
            .bq = 929255560ULL,
            .wk = 923750536ULL,
            .bk = 923750024ULL,
            .wv = 933977736ULL,
            .bv = 933977224ULL,
            .wo = 924536968ULL,
            .wgate = 868696712ULL,
            .wup = 896221832ULL,
            .wdown = 841171592ULL,
        },
        // Camada 13
        {
            .norm1 = 934764168ULL,
            .norm2 = 1017342600ULL,
            .wq = 1022854280ULL,
            .bq = 1022851208ULL,
            .wk = 1017346184ULL,
            .bk = 1017345672ULL,
            .wv = 1027573384ULL,
            .bv = 1027572872ULL,
            .wo = 1018132616ULL,
            .wgate = 962292360ULL,
            .wup = 989817480ULL,
            .wdown = 934767240ULL,
        },
        // Camada 14
        {
            .norm1 = 1028359816ULL,
            .norm2 = 1110938248ULL,
            .wq = 1116449928ULL,
            .bq = 1116446856ULL,
            .wk = 1110941832ULL,
            .bk = 1110941320ULL,
            .wv = 1121169032ULL,
            .bv = 1121168520ULL,
            .wo = 1111728264ULL,
            .wgate = 1055888008ULL,
            .wup = 1083413128ULL,
            .wdown = 1028362888ULL,
        },
        // Camada 15
        {
            .norm1 = 1121955464ULL,
            .norm2 = 1204533896ULL,
            .wq = 1210045576ULL,
            .bq = 1210042504ULL,
            .wk = 1204537480ULL,
            .bk = 1204536968ULL,
            .wv = 1214764680ULL,
            .bv = 1214764168ULL,
            .wo = 1205323912ULL,
            .wgate = 1149483656ULL,
            .wup = 1177008776ULL,
            .wdown = 1121958536ULL,
        },
        // Camada 16
        {
            .norm1 = 1215551112ULL,
            .norm2 = 1298129544ULL,
            .wq = 1303641224ULL,
            .bq = 1303638152ULL,
            .wk = 1298133128ULL,
            .bk = 1298132616ULL,
            .wv = 1308360328ULL,
            .bv = 1308359816ULL,
            .wo = 1298919560ULL,
            .wgate = 1243079304ULL,
            .wup = 1270604424ULL,
            .wdown = 1215554184ULL,
        },
        // Camada 17
        {
            .norm1 = 1309146760ULL,
            .norm2 = 1391725192ULL,
            .wq = 1397236872ULL,
            .bq = 1397233800ULL,
            .wk = 1391728776ULL,
            .bk = 1391728264ULL,
            .wv = 1401955976ULL,
            .bv = 1401955464ULL,
            .wo = 1392515208ULL,
            .wgate = 1336674952ULL,
            .wup = 1364200072ULL,
            .wdown = 1309149832ULL,
        },
        // Camada 18
        {
            .norm1 = 1402742408ULL,
            .norm2 = 1485320840ULL,
            .wq = 1490832520ULL,
            .bq = 1490829448ULL,
            .wk = 1485324424ULL,
            .bk = 1485323912ULL,
            .wv = 1495551624ULL,
            .bv = 1495551112ULL,
            .wo = 1486110856ULL,
            .wgate = 1430270600ULL,
            .wup = 1457795720ULL,
            .wdown = 1402745480ULL,
        },
        // Camada 19
        {
            .norm1 = 1496338056ULL,
            .norm2 = 1578916488ULL,
            .wq = 1584428168ULL,
            .bq = 1584425096ULL,
            .wk = 1578920072ULL,
            .bk = 1578919560ULL,
            .wv = 1589147272ULL,
            .bv = 1589146760ULL,
            .wo = 1579706504ULL,
            .wgate = 1523866248ULL,
            .wup = 1551391368ULL,
            .wdown = 1496341128ULL,
        },
        // Camada 20
        {
            .norm1 = 1683529352ULL,
            .norm2 = 1766107784ULL,
            .wq = 1771619464ULL,
            .bq = 1771616392ULL,
            .wk = 1766111368ULL,
            .bk = 1766110856ULL,
            .wv = 1776338568ULL,
            .bv = 1776338056ULL,
            .wo = 1766897800ULL,
            .wgate = 1711057544ULL,
            .wup = 1738582664ULL,
            .wdown = 1683532424ULL,
        },
        // Camada 21
        {
            .norm1 = 1777125000ULL,
            .norm2 = 1859703432ULL,
            .wq = 1865215112ULL,
            .bq = 1865212040ULL,
            .wk = 1859707016ULL,
            .bk = 1859706504ULL,
            .wv = 1869934216ULL,
            .bv = 1869933704ULL,
            .wo = 1860493448ULL,
            .wgate = 1804653192ULL,
            .wup = 1832178312ULL,
            .wdown = 1777128072ULL,
        },
        // Camada 22
        {
            .norm1 = 1870720648ULL,
            .norm2 = 1953299080ULL,
            .wq = 1958810760ULL,
            .bq = 1958807688ULL,
            .wk = 1953302664ULL,
            .bk = 1953302152ULL,
            .wv = 1963529864ULL,
            .bv = 1963529352ULL,
            .wo = 1954089096ULL,
            .wgate = 1898248840ULL,
            .wup = 1925773960ULL,
            .wdown = 1870723720ULL,
        },
        // Camada 23
        {
            .norm1 = 1964316296ULL,
            .norm2 = 2046894728ULL,
            .wq = 2052406408ULL,
            .bq = 2052403336ULL,
            .wk = 2046898312ULL,
            .bk = 2046897800ULL,
            .wv = 2057125512ULL,
            .bv = 2057125000ULL,
            .wo = 2047684744ULL,
            .wgate = 1991844488ULL,
            .wup = 2019369608ULL,
            .wdown = 1964319368ULL,
        },
        // Camada 24
        {
            .norm1 = 2057911944ULL,
            .norm2 = 2140490376ULL,
            .wq = 2146002056ULL,
            .bq = 2145998984ULL,
            .wk = 2140493960ULL,
            .bk = 2140493448ULL,
            .wv = 2150721160ULL,
            .bv = 2150720648ULL,
            .wo = 2141280392ULL,
            .wgate = 2085440136ULL,
            .wup = 2112965256ULL,
            .wdown = 2057915016ULL,
        },
        // Camada 25
        {
            .norm1 = 2151507592ULL,
            .norm2 = 2234086024ULL,
            .wq = 2239597704ULL,
            .bq = 2239594632ULL,
            .wk = 2234089608ULL,
            .bk = 2234089096ULL,
            .wv = 2244316808ULL,
            .bv = 2244316296ULL,
            .wo = 2234876040ULL,
            .wgate = 2179035784ULL,
            .wup = 2206560904ULL,
            .wdown = 2151510664ULL,
        },
        // Camada 26
        {
            .norm1 = 2245103240ULL,
            .norm2 = 2327681672ULL,
            .wq = 2333193352ULL,
            .bq = 2333190280ULL,
            .wk = 2327685256ULL,
            .bk = 2327684744ULL,
            .wv = 2337912456ULL,
            .bv = 2337911944ULL,
            .wo = 2328471688ULL,
            .wgate = 2272631432ULL,
            .wup = 2300156552ULL,
            .wdown = 2245106312ULL,
        },
        // Camada 27
        {
            .norm1 = 2338698888ULL,
            .norm2 = 2421277320ULL,
            .wq = 2426789000ULL,
            .bq = 2426785928ULL,
            .wk = 2421280904ULL,
            .bk = 2421280392ULL,
            .wv = 2431508104ULL,
            .bv = 2431507592ULL,
            .wo = 2422067336ULL,
            .wgate = 2366227080ULL,
            .wup = 2393752200ULL,
            .wdown = 2338701960ULL,
        }
    }
};

#endif // QWEN_OFFSETS_1_5B_H
