#ifndef QWEN_OFFSETS_1_5B_Q4_H
#define QWEN_OFFSETS_1_5B_Q4_H

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
    .embed_tokens = 0ULL,
    .norm = 131272704ULL,
    .layers = {
        // Layer 0
        {
            .norm1 = 131275776ULL,
            .norm2 = 131278848ULL,
            .wq = 131286016ULL,
            .bq = 131281920ULL,
            .wk = 132613120ULL,
            .bk = 131284992ULL,
            .wv = 132834304ULL,
            .bv = 131285504ULL,
            .wo = 133055488ULL,
            .wgate = 134382592ULL,
            .wup = 142124032ULL,
            .wdown = 149865472ULL,
        },
        // Layer 1
        {
            .norm1 = 157606912ULL,
            .norm2 = 157609984ULL,
            .wq = 157617152ULL,
            .bq = 157613056ULL,
            .wk = 158944256ULL,
            .bk = 157616128ULL,
            .wv = 159165440ULL,
            .bv = 157616640ULL,
            .wo = 159386624ULL,
            .wgate = 160713728ULL,
            .wup = 168455168ULL,
            .wdown = 176196608ULL,
        },
        // Layer 2
        {
            .norm1 = 183938048ULL,
            .norm2 = 183941120ULL,
            .wq = 183948288ULL,
            .bq = 183944192ULL,
            .wk = 185275392ULL,
            .bk = 183947264ULL,
            .wv = 185496576ULL,
            .bv = 183947776ULL,
            .wo = 185717760ULL,
            .wgate = 187044864ULL,
            .wup = 194786304ULL,
            .wdown = 202527744ULL,
        },
        // Layer 3
        {
            .norm1 = 210269184ULL,
            .norm2 = 210272256ULL,
            .wq = 210279424ULL,
            .bq = 210275328ULL,
            .wk = 211606528ULL,
            .bk = 210278400ULL,
            .wv = 211827712ULL,
            .bv = 210278912ULL,
            .wo = 212048896ULL,
            .wgate = 213376000ULL,
            .wup = 221117440ULL,
            .wdown = 228858880ULL,
        },
        // Layer 4
        {
            .norm1 = 236600320ULL,
            .norm2 = 236603392ULL,
            .wq = 236610560ULL,
            .bq = 236606464ULL,
            .wk = 237937664ULL,
            .bk = 236609536ULL,
            .wv = 238158848ULL,
            .bv = 236610048ULL,
            .wo = 238380032ULL,
            .wgate = 239707136ULL,
            .wup = 247448576ULL,
            .wdown = 255190016ULL,
        },
        // Layer 5
        {
            .norm1 = 262931456ULL,
            .norm2 = 262934528ULL,
            .wq = 262941696ULL,
            .bq = 262937600ULL,
            .wk = 264268800ULL,
            .bk = 262940672ULL,
            .wv = 264489984ULL,
            .bv = 262941184ULL,
            .wo = 264711168ULL,
            .wgate = 266038272ULL,
            .wup = 273779712ULL,
            .wdown = 281521152ULL,
        },
        // Layer 6
        {
            .norm1 = 289262592ULL,
            .norm2 = 289265664ULL,
            .wq = 289272832ULL,
            .bq = 289268736ULL,
            .wk = 290599936ULL,
            .bk = 289271808ULL,
            .wv = 290821120ULL,
            .bv = 289272320ULL,
            .wo = 291042304ULL,
            .wgate = 292369408ULL,
            .wup = 300110848ULL,
            .wdown = 307852288ULL,
        },
        // Layer 7
        {
            .norm1 = 315593728ULL,
            .norm2 = 315596800ULL,
            .wq = 315603968ULL,
            .bq = 315599872ULL,
            .wk = 316931072ULL,
            .bk = 315602944ULL,
            .wv = 317152256ULL,
            .bv = 315603456ULL,
            .wo = 317373440ULL,
            .wgate = 318700544ULL,
            .wup = 326441984ULL,
            .wdown = 334183424ULL,
        },
        // Layer 8
        {
            .norm1 = 341924864ULL,
            .norm2 = 341927936ULL,
            .wq = 341935104ULL,
            .bq = 341931008ULL,
            .wk = 343262208ULL,
            .bk = 341934080ULL,
            .wv = 343483392ULL,
            .bv = 341934592ULL,
            .wo = 343704576ULL,
            .wgate = 345031680ULL,
            .wup = 352773120ULL,
            .wdown = 360514560ULL,
        },
        // Layer 9
        {
            .norm1 = 368256000ULL,
            .norm2 = 368259072ULL,
            .wq = 368266240ULL,
            .bq = 368262144ULL,
            .wk = 369593344ULL,
            .bk = 368265216ULL,
            .wv = 369814528ULL,
            .bv = 368265728ULL,
            .wo = 370035712ULL,
            .wgate = 371362816ULL,
            .wup = 379104256ULL,
            .wdown = 386845696ULL,
        },
        // Layer 10
        {
            .norm1 = 394587136ULL,
            .norm2 = 394590208ULL,
            .wq = 394597376ULL,
            .bq = 394593280ULL,
            .wk = 395924480ULL,
            .bk = 394596352ULL,
            .wv = 396145664ULL,
            .bv = 394596864ULL,
            .wo = 396366848ULL,
            .wgate = 397693952ULL,
            .wup = 405435392ULL,
            .wdown = 413176832ULL,
        },
        // Layer 11
        {
            .norm1 = 420918272ULL,
            .norm2 = 420921344ULL,
            .wq = 420928512ULL,
            .bq = 420924416ULL,
            .wk = 422255616ULL,
            .bk = 420927488ULL,
            .wv = 422476800ULL,
            .bv = 420928000ULL,
            .wo = 422697984ULL,
            .wgate = 424025088ULL,
            .wup = 431766528ULL,
            .wdown = 439507968ULL,
        },
        // Layer 12
        {
            .norm1 = 447249408ULL,
            .norm2 = 447252480ULL,
            .wq = 447259648ULL,
            .bq = 447255552ULL,
            .wk = 448586752ULL,
            .bk = 447258624ULL,
            .wv = 448807936ULL,
            .bv = 447259136ULL,
            .wo = 449029120ULL,
            .wgate = 450356224ULL,
            .wup = 458097664ULL,
            .wdown = 465839104ULL,
        },
        // Layer 13
        {
            .norm1 = 473580544ULL,
            .norm2 = 473583616ULL,
            .wq = 473590784ULL,
            .bq = 473586688ULL,
            .wk = 474917888ULL,
            .bk = 473589760ULL,
            .wv = 475139072ULL,
            .bv = 473590272ULL,
            .wo = 475360256ULL,
            .wgate = 476687360ULL,
            .wup = 484428800ULL,
            .wdown = 492170240ULL,
        },
        // Layer 14
        {
            .norm1 = 499911680ULL,
            .norm2 = 499914752ULL,
            .wq = 499921920ULL,
            .bq = 499917824ULL,
            .wk = 501249024ULL,
            .bk = 499920896ULL,
            .wv = 501470208ULL,
            .bv = 499921408ULL,
            .wo = 501691392ULL,
            .wgate = 503018496ULL,
            .wup = 510759936ULL,
            .wdown = 518501376ULL,
        },
        // Layer 15
        {
            .norm1 = 526242816ULL,
            .norm2 = 526245888ULL,
            .wq = 526253056ULL,
            .bq = 526248960ULL,
            .wk = 527580160ULL,
            .bk = 526252032ULL,
            .wv = 527801344ULL,
            .bv = 526252544ULL,
            .wo = 528022528ULL,
            .wgate = 529349632ULL,
            .wup = 537091072ULL,
            .wdown = 544832512ULL,
        },
        // Layer 16
        {
            .norm1 = 552573952ULL,
            .norm2 = 552577024ULL,
            .wq = 552584192ULL,
            .bq = 552580096ULL,
            .wk = 553911296ULL,
            .bk = 552583168ULL,
            .wv = 554132480ULL,
            .bv = 552583680ULL,
            .wo = 554353664ULL,
            .wgate = 555680768ULL,
            .wup = 563422208ULL,
            .wdown = 571163648ULL,
        },
        // Layer 17
        {
            .norm1 = 578905088ULL,
            .norm2 = 578908160ULL,
            .wq = 578915328ULL,
            .bq = 578911232ULL,
            .wk = 580242432ULL,
            .bk = 578914304ULL,
            .wv = 580463616ULL,
            .bv = 578914816ULL,
            .wo = 580684800ULL,
            .wgate = 582011904ULL,
            .wup = 589753344ULL,
            .wdown = 597494784ULL,
        },
        // Layer 18
        {
            .norm1 = 605236224ULL,
            .norm2 = 605239296ULL,
            .wq = 605246464ULL,
            .bq = 605242368ULL,
            .wk = 606573568ULL,
            .bk = 605245440ULL,
            .wv = 606794752ULL,
            .bv = 605245952ULL,
            .wo = 607015936ULL,
            .wgate = 608343040ULL,
            .wup = 616084480ULL,
            .wdown = 623825920ULL,
        },
        // Layer 19
        {
            .norm1 = 631567360ULL,
            .norm2 = 631570432ULL,
            .wq = 631577600ULL,
            .bq = 631573504ULL,
            .wk = 632904704ULL,
            .bk = 631576576ULL,
            .wv = 633125888ULL,
            .bv = 631577088ULL,
            .wo = 633347072ULL,
            .wgate = 634674176ULL,
            .wup = 642415616ULL,
            .wdown = 650157056ULL,
        },
        // Layer 20
        {
            .norm1 = 657898496ULL,
            .norm2 = 657901568ULL,
            .wq = 657908736ULL,
            .bq = 657904640ULL,
            .wk = 659235840ULL,
            .bk = 657907712ULL,
            .wv = 659457024ULL,
            .bv = 657908224ULL,
            .wo = 659678208ULL,
            .wgate = 661005312ULL,
            .wup = 668746752ULL,
            .wdown = 676488192ULL,
        },
        // Layer 21
        {
            .norm1 = 684229632ULL,
            .norm2 = 684232704ULL,
            .wq = 684239872ULL,
            .bq = 684235776ULL,
            .wk = 685566976ULL,
            .bk = 684238848ULL,
            .wv = 685788160ULL,
            .bv = 684239360ULL,
            .wo = 686009344ULL,
            .wgate = 687336448ULL,
            .wup = 695077888ULL,
            .wdown = 702819328ULL,
        },
        // Layer 22
        {
            .norm1 = 710560768ULL,
            .norm2 = 710563840ULL,
            .wq = 710571008ULL,
            .bq = 710566912ULL,
            .wk = 711898112ULL,
            .bk = 710569984ULL,
            .wv = 712119296ULL,
            .bv = 710570496ULL,
            .wo = 712340480ULL,
            .wgate = 713667584ULL,
            .wup = 721409024ULL,
            .wdown = 729150464ULL,
        },
        // Layer 23
        {
            .norm1 = 736891904ULL,
            .norm2 = 736894976ULL,
            .wq = 736902144ULL,
            .bq = 736898048ULL,
            .wk = 738229248ULL,
            .bk = 736901120ULL,
            .wv = 738450432ULL,
            .bv = 736901632ULL,
            .wo = 738671616ULL,
            .wgate = 739998720ULL,
            .wup = 747740160ULL,
            .wdown = 755481600ULL,
        },
        // Layer 24
        {
            .norm1 = 763223040ULL,
            .norm2 = 763226112ULL,
            .wq = 763233280ULL,
            .bq = 763229184ULL,
            .wk = 764560384ULL,
            .bk = 763232256ULL,
            .wv = 764781568ULL,
            .bv = 763232768ULL,
            .wo = 765002752ULL,
            .wgate = 766329856ULL,
            .wup = 774071296ULL,
            .wdown = 781812736ULL,
        },
        // Layer 25
        {
            .norm1 = 789554176ULL,
            .norm2 = 789557248ULL,
            .wq = 789564416ULL,
            .bq = 789560320ULL,
            .wk = 790891520ULL,
            .bk = 789563392ULL,
            .wv = 791112704ULL,
            .bv = 789563904ULL,
            .wo = 791333888ULL,
            .wgate = 792660992ULL,
            .wup = 800402432ULL,
            .wdown = 808143872ULL,
        },
        // Layer 26
        {
            .norm1 = 815885312ULL,
            .norm2 = 815888384ULL,
            .wq = 815895552ULL,
            .bq = 815891456ULL,
            .wk = 817222656ULL,
            .bk = 815894528ULL,
            .wv = 817443840ULL,
            .bv = 815895040ULL,
            .wo = 817665024ULL,
            .wgate = 818992128ULL,
            .wup = 826733568ULL,
            .wdown = 834475008ULL,
        },
        // Layer 27
        {
            .norm1 = 842216448ULL,
            .norm2 = 842219520ULL,
            .wq = 842226688ULL,
            .bq = 842222592ULL,
            .wk = 843553792ULL,
            .bk = 842225664ULL,
            .wv = 843774976ULL,
            .bv = 842226176ULL,
            .wo = 843996160ULL,
            .wgate = 845323264ULL,
            .wup = 853064704ULL,
            .wdown = 860806144ULL,
        }
    }
};

#endif // QWEN_OFFSETS_1_5B_Q4_H
