#ifndef QWEN_OFFSETS_H
#define QWEN_OFFSETS_H

#include <stdint.h>

#define QWEN_VOCAB_SIZE 151936
#define QWEN_DIM 896
#define QWEN_INTERMEDIATE_DIM 4864
#define QWEN_NUM_LAYERS 24
#define QWEN_NUM_HEADS 14
#define QWEN_NUM_KV_HEADS 2
#define QWEN_HEAD_DIM 64
#define QWEN_ROPE_THETA 1000000.0f
#define QWEN_RMS_EPS 1e-6f

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
} QwenLayerOffsets;

typedef struct {
    uint64_t embed_tokens;
    uint64_t norm;
    QwenLayerOffsets layers[QWEN_NUM_LAYERS];
} QwenModelOffsets;

static const QwenModelOffsets g_qwen_offsets = {
    .embed_tokens = 32288ULL,
    .norm = 988096032ULL,
    .layers = {
        // Camada 0
        {
            .norm1 = 272301600ULL,
            .norm2 = 298452256ULL,
            .wq = 300291104ULL,
            .bq = 300289312ULL,
            .wk = 298454304ULL,
            .bk = 298454048ULL,
            .wv = 301896992ULL,
            .bv = 301896736ULL,
            .wo = 298683680ULL,
            .wgate = 281019680ULL,
            .wup = 289735968ULL,
            .wdown = 272303392ULL,
        },
        // Camada 1
        {
            .norm1 = 302126368ULL,
            .norm2 = 328277024ULL,
            .wq = 330115872ULL,
            .bq = 330114080ULL,
            .wk = 328279072ULL,
            .bk = 328278816ULL,
            .wv = 331721760ULL,
            .bv = 331721504ULL,
            .wo = 328508448ULL,
            .wgate = 310844448ULL,
            .wup = 319560736ULL,
            .wdown = 302128160ULL,
        },
        // Camada 2
        {
            .norm1 = 630198816ULL,
            .norm2 = 656349472ULL,
            .wq = 658188320ULL,
            .bq = 658186528ULL,
            .wk = 656351520ULL,
            .bk = 656351264ULL,
            .wv = 659794208ULL,
            .bv = 659793952ULL,
            .wo = 656580896ULL,
            .wgate = 638916896ULL,
            .wup = 647633184ULL,
            .wdown = 630200608ULL,
        },
        // Camada 3
        {
            .norm1 = 779322656ULL,
            .norm2 = 805473312ULL,
            .wq = 807312160ULL,
            .bq = 807310368ULL,
            .wk = 805475360ULL,
            .bk = 805475104ULL,
            .wv = 808918048ULL,
            .bv = 808917792ULL,
            .wo = 805704736ULL,
            .wgate = 788040736ULL,
            .wup = 796757024ULL,
            .wdown = 779324448ULL,
        },
        // Camada 4
        {
            .norm1 = 809147424ULL,
            .norm2 = 835298080ULL,
            .wq = 837136928ULL,
            .bq = 837135136ULL,
            .wk = 835300128ULL,
            .bk = 835299872ULL,
            .wv = 838742816ULL,
            .bv = 838742560ULL,
            .wo = 835529504ULL,
            .wgate = 817865504ULL,
            .wup = 826581792ULL,
            .wdown = 809149216ULL,
        },
        // Camada 5
        {
            .norm1 = 838972192ULL,
            .norm2 = 865122848ULL,
            .wq = 866961696ULL,
            .bq = 866959904ULL,
            .wk = 865124896ULL,
            .bk = 865124640ULL,
            .wv = 868567584ULL,
            .bv = 868567328ULL,
            .wo = 865354272ULL,
            .wgate = 847690272ULL,
            .wup = 856406560ULL,
            .wdown = 838973984ULL,
        },
        // Camada 6
        {
            .norm1 = 868796960ULL,
            .norm2 = 894947616ULL,
            .wq = 896786464ULL,
            .bq = 896784672ULL,
            .wk = 894949664ULL,
            .bk = 894949408ULL,
            .wv = 898392352ULL,
            .bv = 898392096ULL,
            .wo = 895179040ULL,
            .wgate = 877515040ULL,
            .wup = 886231328ULL,
            .wdown = 868798752ULL,
        },
        // Camada 7
        {
            .norm1 = 898621728ULL,
            .norm2 = 924772384ULL,
            .wq = 926611232ULL,
            .bq = 926609440ULL,
            .wk = 924774432ULL,
            .bk = 924774176ULL,
            .wv = 928217120ULL,
            .bv = 928216864ULL,
            .wo = 925003808ULL,
            .wgate = 907339808ULL,
            .wup = 916056096ULL,
            .wdown = 898623520ULL,
        },
        // Camada 8
        {
            .norm1 = 928446496ULL,
            .norm2 = 954597152ULL,
            .wq = 956436000ULL,
            .bq = 956434208ULL,
            .wk = 954599200ULL,
            .bk = 954598944ULL,
            .wv = 958041888ULL,
            .bv = 958041632ULL,
            .wo = 954828576ULL,
            .wgate = 937164576ULL,
            .wup = 945880864ULL,
            .wdown = 928448288ULL,
        },
        // Camada 9
        {
            .norm1 = 958271264ULL,
            .norm2 = 984421920ULL,
            .wq = 986260768ULL,
            .bq = 986258976ULL,
            .wk = 984423968ULL,
            .bk = 984423712ULL,
            .wv = 987866656ULL,
            .bv = 987866400ULL,
            .wo = 984653344ULL,
            .wgate = 966989344ULL,
            .wup = 975705632ULL,
            .wdown = 958273056ULL,
        },
        // Camada 10
        {
            .norm1 = 331951136ULL,
            .norm2 = 358101792ULL,
            .wq = 359940640ULL,
            .bq = 359938848ULL,
            .wk = 358103840ULL,
            .bk = 358103584ULL,
            .wv = 361546528ULL,
            .bv = 361546272ULL,
            .wo = 358333216ULL,
            .wgate = 340669216ULL,
            .wup = 349385504ULL,
            .wdown = 331952928ULL,
        },
        // Camada 11
        {
            .norm1 = 361775904ULL,
            .norm2 = 387926560ULL,
            .wq = 389765408ULL,
            .bq = 389763616ULL,
            .wk = 387928608ULL,
            .bk = 387928352ULL,
            .wv = 391371296ULL,
            .bv = 391371040ULL,
            .wo = 388157984ULL,
            .wgate = 370493984ULL,
            .wup = 379210272ULL,
            .wdown = 361777696ULL,
        },
        // Camada 12
        {
            .norm1 = 391600672ULL,
            .norm2 = 417751328ULL,
            .wq = 419590176ULL,
            .bq = 419588384ULL,
            .wk = 417753376ULL,
            .bk = 417753120ULL,
            .wv = 421196064ULL,
            .bv = 421195808ULL,
            .wo = 417982752ULL,
            .wgate = 400318752ULL,
            .wup = 409035040ULL,
            .wdown = 391602464ULL,
        },
        // Camada 13
        {
            .norm1 = 421425440ULL,
            .norm2 = 447576096ULL,
            .wq = 449414944ULL,
            .bq = 449413152ULL,
            .wk = 447578144ULL,
            .bk = 447577888ULL,
            .wv = 451020832ULL,
            .bv = 451020576ULL,
            .wo = 447807520ULL,
            .wgate = 430143520ULL,
            .wup = 438859808ULL,
            .wdown = 421427232ULL,
        },
        // Camada 14
        {
            .norm1 = 451250208ULL,
            .norm2 = 477400864ULL,
            .wq = 479239712ULL,
            .bq = 479237920ULL,
            .wk = 477402912ULL,
            .bk = 477402656ULL,
            .wv = 480845600ULL,
            .bv = 480845344ULL,
            .wo = 477632288ULL,
            .wgate = 459968288ULL,
            .wup = 468684576ULL,
            .wdown = 451252000ULL,
        },
        // Camada 15
        {
            .norm1 = 481074976ULL,
            .norm2 = 507225632ULL,
            .wq = 509064480ULL,
            .bq = 509062688ULL,
            .wk = 507227680ULL,
            .bk = 507227424ULL,
            .wv = 510670368ULL,
            .bv = 510670112ULL,
            .wo = 507457056ULL,
            .wgate = 489793056ULL,
            .wup = 498509344ULL,
            .wdown = 481076768ULL,
        },
        // Camada 16
        {
            .norm1 = 510899744ULL,
            .norm2 = 537050400ULL,
            .wq = 538889248ULL,
            .bq = 538887456ULL,
            .wk = 537052448ULL,
            .bk = 537052192ULL,
            .wv = 540495136ULL,
            .bv = 540494880ULL,
            .wo = 537281824ULL,
            .wgate = 519617824ULL,
            .wup = 528334112ULL,
            .wdown = 510901536ULL,
        },
        // Camada 17
        {
            .norm1 = 540724512ULL,
            .norm2 = 566875168ULL,
            .wq = 568714016ULL,
            .bq = 568712224ULL,
            .wk = 566877216ULL,
            .bk = 566876960ULL,
            .wv = 570319904ULL,
            .bv = 570319648ULL,
            .wo = 567106592ULL,
            .wgate = 549442592ULL,
            .wup = 558158880ULL,
            .wdown = 540726304ULL,
        },
        // Camada 18
        {
            .norm1 = 570549280ULL,
            .norm2 = 596699936ULL,
            .wq = 598538784ULL,
            .bq = 598536992ULL,
            .wk = 596701984ULL,
            .bk = 596701728ULL,
            .wv = 600144672ULL,
            .bv = 600144416ULL,
            .wo = 596931360ULL,
            .wgate = 579267360ULL,
            .wup = 587983648ULL,
            .wdown = 570551072ULL,
        },
        // Camada 19
        {
            .norm1 = 600374048ULL,
            .norm2 = 626524704ULL,
            .wq = 628363552ULL,
            .bq = 628361760ULL,
            .wk = 626526752ULL,
            .bk = 626526496ULL,
            .wv = 629969440ULL,
            .bv = 629969184ULL,
            .wo = 626756128ULL,
            .wgate = 609092128ULL,
            .wup = 617808416ULL,
            .wdown = 600375840ULL,
        },
        // Camada 20
        {
            .norm1 = 660023584ULL,
            .norm2 = 686174240ULL,
            .wq = 688013088ULL,
            .bq = 688011296ULL,
            .wk = 686176288ULL,
            .bk = 686176032ULL,
            .wv = 689618976ULL,
            .bv = 689618720ULL,
            .wo = 686405664ULL,
            .wgate = 668741664ULL,
            .wup = 677457952ULL,
            .wdown = 660025376ULL,
        },
        // Camada 21
        {
            .norm1 = 689848352ULL,
            .norm2 = 715999008ULL,
            .wq = 717837856ULL,
            .bq = 717836064ULL,
            .wk = 716001056ULL,
            .bk = 716000800ULL,
            .wv = 719443744ULL,
            .bv = 719443488ULL,
            .wo = 716230432ULL,
            .wgate = 698566432ULL,
            .wup = 707282720ULL,
            .wdown = 689850144ULL,
        },
        // Camada 22
        {
            .norm1 = 719673120ULL,
            .norm2 = 745823776ULL,
            .wq = 747662624ULL,
            .bq = 747660832ULL,
            .wk = 745825824ULL,
            .bk = 745825568ULL,
            .wv = 749268512ULL,
            .bv = 749268256ULL,
            .wo = 746055200ULL,
            .wgate = 728391200ULL,
            .wup = 737107488ULL,
            .wdown = 719674912ULL,
        },
        // Camada 23
        {
            .norm1 = 749497888ULL,
            .norm2 = 775648544ULL,
            .wq = 777487392ULL,
            .bq = 777485600ULL,
            .wk = 775650592ULL,
            .bk = 775650336ULL,
            .wv = 779093280ULL,
            .bv = 779093024ULL,
            .wo = 775879968ULL,
            .wgate = 758215968ULL,
            .wup = 766932256ULL,
            .wdown = 749499680ULL,
        }
    }
};

#endif // QWEN_OFFSETS_H
