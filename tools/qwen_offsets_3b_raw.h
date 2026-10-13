#ifndef QWEN_OFFSETS_3B_RAW_H
#define QWEN_OFFSETS_3B_RAW_H

#include <stdint.h>

typedef struct {
    uint8_t shard;
    uint64_t offset;
} RawTensorLoc;

typedef struct {
    RawTensorLoc norm1;
    RawTensorLoc norm2;
    RawTensorLoc wq;
    RawTensorLoc bq;
    RawTensorLoc wk;
    RawTensorLoc bk;
    RawTensorLoc wv;
    RawTensorLoc bv;
    RawTensorLoc wo;
    RawTensorLoc wgate;
    RawTensorLoc wup;
    RawTensorLoc wdown;
} RawLayerLocs;

typedef struct {
    RawTensorLoc embed_tokens;
    RawTensorLoc norm;
    RawLayerLocs layers[36];
} RawModelLocs;

static const RawModelLocs g_raw_model_locs = {
    .embed_tokens = { .shard = 0, .offset = 30208ULL },
    .norm = { .shard = 1, .offset = 2203263952ULL },
    .layers = {
        // Camada 0
        {
            .norm1 = { .shard = 0, .offset = 622360064ULL },
            .norm2 = { .shard = 0, .offset = 757630464ULL },
            .wq = { .shard = 0, .offset = 767076352ULL },
            .bq = { .shard = 0, .offset = 767072256ULL },
            .wk = { .shard = 0, .offset = 757635072ULL },
            .bk = { .shard = 0, .offset = 757634560ULL },
            .wv = { .shard = 0, .offset = 775465472ULL },
            .bv = { .shard = 0, .offset = 775464960ULL },
            .wo = { .shard = 0, .offset = 758683648ULL },
            .wgate = { .shard = 0, .offset = 667452928ULL },
            .wup = { .shard = 0, .offset = 712541696ULL },
            .wdown = { .shard = 0, .offset = 622364160ULL }
        },
        // Camada 1
        {
            .norm1 = { .shard = 0, .offset = 776514048ULL },
            .norm2 = { .shard = 0, .offset = 911784448ULL },
            .wq = { .shard = 0, .offset = 921230336ULL },
            .bq = { .shard = 0, .offset = 921226240ULL },
            .wk = { .shard = 0, .offset = 911789056ULL },
            .bk = { .shard = 0, .offset = 911788544ULL },
            .wv = { .shard = 0, .offset = 929619456ULL },
            .bv = { .shard = 0, .offset = 929618944ULL },
            .wo = { .shard = 0, .offset = 912837632ULL },
            .wgate = { .shard = 0, .offset = 821606912ULL },
            .wup = { .shard = 0, .offset = 866695680ULL },
            .wdown = { .shard = 0, .offset = 776518144ULL }
        },
        // Camada 2
        {
            .norm1 = { .shard = 0, .offset = 2472207872ULL },
            .norm2 = { .shard = 0, .offset = 2607478272ULL },
            .wq = { .shard = 0, .offset = 2616924160ULL },
            .bq = { .shard = 0, .offset = 2616920064ULL },
            .wk = { .shard = 0, .offset = 2607482880ULL },
            .bk = { .shard = 0, .offset = 2607482368ULL },
            .wv = { .shard = 0, .offset = 2625313280ULL },
            .bv = { .shard = 0, .offset = 2625312768ULL },
            .wo = { .shard = 0, .offset = 2608531456ULL },
            .wgate = { .shard = 0, .offset = 2517300736ULL },
            .wup = { .shard = 0, .offset = 2562389504ULL },
            .wdown = { .shard = 0, .offset = 2472211968ULL }
        },
        // Camada 3
        {
            .norm1 = { .shard = 0, .offset = 2889581056ULL },
            .norm2 = { .shard = 0, .offset = 3024851456ULL },
            .wq = { .shard = 0, .offset = 3034297344ULL },
            .bq = { .shard = 0, .offset = 3034293248ULL },
            .wk = { .shard = 0, .offset = 3024856064ULL },
            .bk = { .shard = 0, .offset = 3024855552ULL },
            .wv = { .shard = 0, .offset = 3042686464ULL },
            .bv = { .shard = 0, .offset = 3042685952ULL },
            .wo = { .shard = 0, .offset = 3025904640ULL },
            .wgate = { .shard = 0, .offset = 2934673920ULL },
            .wup = { .shard = 0, .offset = 2979762688ULL },
            .wdown = { .shard = 0, .offset = 2889585152ULL }
        },
        // Camada 4
        {
            .norm1 = { .shard = 0, .offset = 3043735040ULL },
            .norm2 = { .shard = 0, .offset = 3179005440ULL },
            .wq = { .shard = 0, .offset = 3188451328ULL },
            .bq = { .shard = 0, .offset = 3188447232ULL },
            .wk = { .shard = 0, .offset = 3179010048ULL },
            .bk = { .shard = 0, .offset = 3179009536ULL },
            .wv = { .shard = 0, .offset = 3196840448ULL },
            .bv = { .shard = 0, .offset = 3196839936ULL },
            .wo = { .shard = 0, .offset = 3180058624ULL },
            .wgate = { .shard = 0, .offset = 3088827904ULL },
            .wup = { .shard = 0, .offset = 3133916672ULL },
            .wdown = { .shard = 0, .offset = 3043739136ULL }
        },
        // Camada 5
        {
            .norm1 = { .shard = 0, .offset = 3197889024ULL },
            .norm2 = { .shard = 0, .offset = 3333159424ULL },
            .wq = { .shard = 0, .offset = 3342605312ULL },
            .bq = { .shard = 0, .offset = 3342601216ULL },
            .wk = { .shard = 0, .offset = 3333164032ULL },
            .bk = { .shard = 0, .offset = 3333163520ULL },
            .wv = { .shard = 0, .offset = 3350994432ULL },
            .bv = { .shard = 0, .offset = 3350993920ULL },
            .wo = { .shard = 0, .offset = 3334212608ULL },
            .wgate = { .shard = 0, .offset = 3242981888ULL },
            .wup = { .shard = 0, .offset = 3288070656ULL },
            .wdown = { .shard = 0, .offset = 3197893120ULL }
        },
        // Camada 6
        {
            .norm1 = { .shard = 0, .offset = 3352043008ULL },
            .norm2 = { .shard = 0, .offset = 3487313408ULL },
            .wq = { .shard = 0, .offset = 3496759296ULL },
            .bq = { .shard = 0, .offset = 3496755200ULL },
            .wk = { .shard = 0, .offset = 3487318016ULL },
            .bk = { .shard = 0, .offset = 3487317504ULL },
            .wv = { .shard = 0, .offset = 3505148416ULL },
            .bv = { .shard = 0, .offset = 3505147904ULL },
            .wo = { .shard = 0, .offset = 3488366592ULL },
            .wgate = { .shard = 0, .offset = 3397135872ULL },
            .wup = { .shard = 0, .offset = 3442224640ULL },
            .wdown = { .shard = 0, .offset = 3352047104ULL }
        },
        // Camada 7
        {
            .norm1 = { .shard = 0, .offset = 3506196992ULL },
            .norm2 = { .shard = 0, .offset = 3641467392ULL },
            .wq = { .shard = 0, .offset = 3650913280ULL },
            .bq = { .shard = 0, .offset = 3650909184ULL },
            .wk = { .shard = 0, .offset = 3641472000ULL },
            .bk = { .shard = 0, .offset = 3641471488ULL },
            .wv = { .shard = 0, .offset = 3659302400ULL },
            .bv = { .shard = 0, .offset = 3659301888ULL },
            .wo = { .shard = 0, .offset = 3642520576ULL },
            .wgate = { .shard = 0, .offset = 3551289856ULL },
            .wup = { .shard = 0, .offset = 3596378624ULL },
            .wdown = { .shard = 0, .offset = 3506201088ULL }
        },
        // Camada 8
        {
            .norm1 = { .shard = 0, .offset = 3660350976ULL },
            .norm2 = { .shard = 0, .offset = 3795621376ULL },
            .wq = { .shard = 0, .offset = 3805067264ULL },
            .bq = { .shard = 0, .offset = 3805063168ULL },
            .wk = { .shard = 0, .offset = 3795625984ULL },
            .bk = { .shard = 0, .offset = 3795625472ULL },
            .wv = { .shard = 0, .offset = 3813456384ULL },
            .bv = { .shard = 0, .offset = 3813455872ULL },
            .wo = { .shard = 0, .offset = 3796674560ULL },
            .wgate = { .shard = 0, .offset = 3705443840ULL },
            .wup = { .shard = 0, .offset = 3750532608ULL },
            .wdown = { .shard = 0, .offset = 3660355072ULL }
        },
        // Camada 9
        {
            .norm1 = { .shard = 0, .offset = 3814504960ULL },
            .norm2 = { .shard = 0, .offset = 3949775360ULL },
            .wq = { .shard = 0, .offset = 3959221248ULL },
            .bq = { .shard = 0, .offset = 3959217152ULL },
            .wk = { .shard = 0, .offset = 3949779968ULL },
            .bk = { .shard = 0, .offset = 3949779456ULL },
            .wv = { .shard = 0, .offset = 3967610368ULL },
            .bv = { .shard = 0, .offset = 3967609856ULL },
            .wo = { .shard = 0, .offset = 3950828544ULL },
            .wgate = { .shard = 0, .offset = 3859597824ULL },
            .wup = { .shard = 0, .offset = 3904686592ULL },
            .wdown = { .shard = 0, .offset = 3814509056ULL }
        },
        // Camada 10
        {
            .norm1 = { .shard = 0, .offset = 930668032ULL },
            .norm2 = { .shard = 0, .offset = 1065938432ULL },
            .wq = { .shard = 0, .offset = 1075384320ULL },
            .bq = { .shard = 0, .offset = 1075380224ULL },
            .wk = { .shard = 0, .offset = 1065943040ULL },
            .bk = { .shard = 0, .offset = 1065942528ULL },
            .wv = { .shard = 0, .offset = 1083773440ULL },
            .bv = { .shard = 0, .offset = 1083772928ULL },
            .wo = { .shard = 0, .offset = 1066991616ULL },
            .wgate = { .shard = 0, .offset = 975760896ULL },
            .wup = { .shard = 0, .offset = 1020849664ULL },
            .wdown = { .shard = 0, .offset = 930672128ULL }
        },
        // Camada 11
        {
            .norm1 = { .shard = 0, .offset = 1084822016ULL },
            .norm2 = { .shard = 0, .offset = 1220092416ULL },
            .wq = { .shard = 0, .offset = 1229538304ULL },
            .bq = { .shard = 0, .offset = 1229534208ULL },
            .wk = { .shard = 0, .offset = 1220097024ULL },
            .bk = { .shard = 0, .offset = 1220096512ULL },
            .wv = { .shard = 0, .offset = 1237927424ULL },
            .bv = { .shard = 0, .offset = 1237926912ULL },
            .wo = { .shard = 0, .offset = 1221145600ULL },
            .wgate = { .shard = 0, .offset = 1129914880ULL },
            .wup = { .shard = 0, .offset = 1175003648ULL },
            .wdown = { .shard = 0, .offset = 1084826112ULL }
        },
        // Camada 12
        {
            .norm1 = { .shard = 0, .offset = 1238976000ULL },
            .norm2 = { .shard = 0, .offset = 1374246400ULL },
            .wq = { .shard = 0, .offset = 1383692288ULL },
            .bq = { .shard = 0, .offset = 1383688192ULL },
            .wk = { .shard = 0, .offset = 1374251008ULL },
            .bk = { .shard = 0, .offset = 1374250496ULL },
            .wv = { .shard = 0, .offset = 1392081408ULL },
            .bv = { .shard = 0, .offset = 1392080896ULL },
            .wo = { .shard = 0, .offset = 1375299584ULL },
            .wgate = { .shard = 0, .offset = 1284068864ULL },
            .wup = { .shard = 0, .offset = 1329157632ULL },
            .wdown = { .shard = 0, .offset = 1238980096ULL }
        },
        // Camada 13
        {
            .norm1 = { .shard = 0, .offset = 1393129984ULL },
            .norm2 = { .shard = 0, .offset = 1528400384ULL },
            .wq = { .shard = 0, .offset = 1537846272ULL },
            .bq = { .shard = 0, .offset = 1537842176ULL },
            .wk = { .shard = 0, .offset = 1528404992ULL },
            .bk = { .shard = 0, .offset = 1528404480ULL },
            .wv = { .shard = 0, .offset = 1546235392ULL },
            .bv = { .shard = 0, .offset = 1546234880ULL },
            .wo = { .shard = 0, .offset = 1529453568ULL },
            .wgate = { .shard = 0, .offset = 1438222848ULL },
            .wup = { .shard = 0, .offset = 1483311616ULL },
            .wdown = { .shard = 0, .offset = 1393134080ULL }
        },
        // Camada 14
        {
            .norm1 = { .shard = 0, .offset = 1547283968ULL },
            .norm2 = { .shard = 0, .offset = 1682554368ULL },
            .wq = { .shard = 0, .offset = 1692000256ULL },
            .bq = { .shard = 0, .offset = 1691996160ULL },
            .wk = { .shard = 0, .offset = 1682558976ULL },
            .bk = { .shard = 0, .offset = 1682558464ULL },
            .wv = { .shard = 0, .offset = 1700389376ULL },
            .bv = { .shard = 0, .offset = 1700388864ULL },
            .wo = { .shard = 0, .offset = 1683607552ULL },
            .wgate = { .shard = 0, .offset = 1592376832ULL },
            .wup = { .shard = 0, .offset = 1637465600ULL },
            .wdown = { .shard = 0, .offset = 1547288064ULL }
        },
        // Camada 15
        {
            .norm1 = { .shard = 0, .offset = 1701437952ULL },
            .norm2 = { .shard = 0, .offset = 1836708352ULL },
            .wq = { .shard = 0, .offset = 1846154240ULL },
            .bq = { .shard = 0, .offset = 1846150144ULL },
            .wk = { .shard = 0, .offset = 1836712960ULL },
            .bk = { .shard = 0, .offset = 1836712448ULL },
            .wv = { .shard = 0, .offset = 1854543360ULL },
            .bv = { .shard = 0, .offset = 1854542848ULL },
            .wo = { .shard = 0, .offset = 1837761536ULL },
            .wgate = { .shard = 0, .offset = 1746530816ULL },
            .wup = { .shard = 0, .offset = 1791619584ULL },
            .wdown = { .shard = 0, .offset = 1701442048ULL }
        },
        // Camada 16
        {
            .norm1 = { .shard = 0, .offset = 1855591936ULL },
            .norm2 = { .shard = 0, .offset = 1990862336ULL },
            .wq = { .shard = 0, .offset = 2000308224ULL },
            .bq = { .shard = 0, .offset = 2000304128ULL },
            .wk = { .shard = 0, .offset = 1990866944ULL },
            .bk = { .shard = 0, .offset = 1990866432ULL },
            .wv = { .shard = 0, .offset = 2008697344ULL },
            .bv = { .shard = 0, .offset = 2008696832ULL },
            .wo = { .shard = 0, .offset = 1991915520ULL },
            .wgate = { .shard = 0, .offset = 1900684800ULL },
            .wup = { .shard = 0, .offset = 1945773568ULL },
            .wdown = { .shard = 0, .offset = 1855596032ULL }
        },
        // Camada 17
        {
            .norm1 = { .shard = 0, .offset = 2009745920ULL },
            .norm2 = { .shard = 0, .offset = 2145016320ULL },
            .wq = { .shard = 0, .offset = 2154462208ULL },
            .bq = { .shard = 0, .offset = 2154458112ULL },
            .wk = { .shard = 0, .offset = 2145020928ULL },
            .bk = { .shard = 0, .offset = 2145020416ULL },
            .wv = { .shard = 0, .offset = 2162851328ULL },
            .bv = { .shard = 0, .offset = 2162850816ULL },
            .wo = { .shard = 0, .offset = 2146069504ULL },
            .wgate = { .shard = 0, .offset = 2054838784ULL },
            .wup = { .shard = 0, .offset = 2099927552ULL },
            .wdown = { .shard = 0, .offset = 2009750016ULL }
        },
        // Camada 18
        {
            .norm1 = { .shard = 0, .offset = 2163899904ULL },
            .norm2 = { .shard = 0, .offset = 2299170304ULL },
            .wq = { .shard = 0, .offset = 2308616192ULL },
            .bq = { .shard = 0, .offset = 2308612096ULL },
            .wk = { .shard = 0, .offset = 2299174912ULL },
            .bk = { .shard = 0, .offset = 2299174400ULL },
            .wv = { .shard = 0, .offset = 2317005312ULL },
            .bv = { .shard = 0, .offset = 2317004800ULL },
            .wo = { .shard = 0, .offset = 2300223488ULL },
            .wgate = { .shard = 0, .offset = 2208992768ULL },
            .wup = { .shard = 0, .offset = 2254081536ULL },
            .wdown = { .shard = 0, .offset = 2163904000ULL }
        },
        // Camada 19
        {
            .norm1 = { .shard = 0, .offset = 2318053888ULL },
            .norm2 = { .shard = 0, .offset = 2453324288ULL },
            .wq = { .shard = 0, .offset = 2462770176ULL },
            .bq = { .shard = 0, .offset = 2462766080ULL },
            .wk = { .shard = 0, .offset = 2453328896ULL },
            .bk = { .shard = 0, .offset = 2453328384ULL },
            .wv = { .shard = 0, .offset = 2471159296ULL },
            .bv = { .shard = 0, .offset = 2471158784ULL },
            .wo = { .shard = 0, .offset = 2454377472ULL },
            .wgate = { .shard = 0, .offset = 2363146752ULL },
            .wup = { .shard = 0, .offset = 2408235520ULL },
            .wdown = { .shard = 0, .offset = 2318057984ULL }
        },
        // Camada 20
        {
            .norm1 = { .shard = 0, .offset = 2626361856ULL },
            .norm2 = { .shard = 0, .offset = 2761632256ULL },
            .wq = { .shard = 0, .offset = 2771078144ULL },
            .bq = { .shard = 0, .offset = 2771074048ULL },
            .wk = { .shard = 0, .offset = 2761636864ULL },
            .bk = { .shard = 0, .offset = 2761636352ULL },
            .wv = { .shard = 0, .offset = 2779467264ULL },
            .bv = { .shard = 0, .offset = 2779466752ULL },
            .wo = { .shard = 0, .offset = 2762685440ULL },
            .wgate = { .shard = 0, .offset = 2671454720ULL },
            .wup = { .shard = 0, .offset = 2716543488ULL },
            .wdown = { .shard = 0, .offset = 2626365952ULL }
        },
        // Camada 21
        {
            .norm1 = { .shard = 0, .offset = 2780515840ULL },
            .norm2 = { .shard = 0, .offset = 2870697472ULL },
            .wq = { .shard = 0, .offset = 2880143360ULL },
            .bq = { .shard = 0, .offset = 2880139264ULL },
            .wk = { .shard = 0, .offset = 2870702080ULL },
            .bk = { .shard = 0, .offset = 2870701568ULL },
            .wv = { .shard = 0, .offset = 2888532480ULL },
            .bv = { .shard = 0, .offset = 2888531968ULL },
            .wo = { .shard = 0, .offset = 2871750656ULL },
            .wgate = { .shard = 0, .offset = 2780519936ULL },
            .wup = { .shard = 0, .offset = 2825608704ULL },
            .wdown = { .shard = 1, .offset = 19408ULL }
        },
        // Camada 22
        {
            .norm1 = { .shard = 1, .offset = 45108176ULL },
            .norm2 = { .shard = 1, .offset = 180378576ULL },
            .wq = { .shard = 1, .offset = 189824464ULL },
            .bq = { .shard = 1, .offset = 189820368ULL },
            .wk = { .shard = 1, .offset = 180383184ULL },
            .bk = { .shard = 1, .offset = 180382672ULL },
            .wv = { .shard = 1, .offset = 198213584ULL },
            .bv = { .shard = 1, .offset = 198213072ULL },
            .wo = { .shard = 1, .offset = 181431760ULL },
            .wgate = { .shard = 1, .offset = 90201040ULL },
            .wup = { .shard = 1, .offset = 135289808ULL },
            .wdown = { .shard = 1, .offset = 45112272ULL }
        },
        // Camada 23
        {
            .norm1 = { .shard = 1, .offset = 199262160ULL },
            .norm2 = { .shard = 1, .offset = 334532560ULL },
            .wq = { .shard = 1, .offset = 343978448ULL },
            .bq = { .shard = 1, .offset = 343974352ULL },
            .wk = { .shard = 1, .offset = 334537168ULL },
            .bk = { .shard = 1, .offset = 334536656ULL },
            .wv = { .shard = 1, .offset = 352367568ULL },
            .bv = { .shard = 1, .offset = 352367056ULL },
            .wo = { .shard = 1, .offset = 335585744ULL },
            .wgate = { .shard = 1, .offset = 244355024ULL },
            .wup = { .shard = 1, .offset = 289443792ULL },
            .wdown = { .shard = 1, .offset = 199266256ULL }
        },
        // Camada 24
        {
            .norm1 = { .shard = 1, .offset = 353416144ULL },
            .norm2 = { .shard = 1, .offset = 488686544ULL },
            .wq = { .shard = 1, .offset = 498132432ULL },
            .bq = { .shard = 1, .offset = 498128336ULL },
            .wk = { .shard = 1, .offset = 488691152ULL },
            .bk = { .shard = 1, .offset = 488690640ULL },
            .wv = { .shard = 1, .offset = 506521552ULL },
            .bv = { .shard = 1, .offset = 506521040ULL },
            .wo = { .shard = 1, .offset = 489739728ULL },
            .wgate = { .shard = 1, .offset = 398509008ULL },
            .wup = { .shard = 1, .offset = 443597776ULL },
            .wdown = { .shard = 1, .offset = 353420240ULL }
        },
        // Camada 25
        {
            .norm1 = { .shard = 1, .offset = 507570128ULL },
            .norm2 = { .shard = 1, .offset = 642840528ULL },
            .wq = { .shard = 1, .offset = 652286416ULL },
            .bq = { .shard = 1, .offset = 652282320ULL },
            .wk = { .shard = 1, .offset = 642845136ULL },
            .bk = { .shard = 1, .offset = 642844624ULL },
            .wv = { .shard = 1, .offset = 660675536ULL },
            .bv = { .shard = 1, .offset = 660675024ULL },
            .wo = { .shard = 1, .offset = 643893712ULL },
            .wgate = { .shard = 1, .offset = 552662992ULL },
            .wup = { .shard = 1, .offset = 597751760ULL },
            .wdown = { .shard = 1, .offset = 507574224ULL }
        },
        // Camada 26
        {
            .norm1 = { .shard = 1, .offset = 661724112ULL },
            .norm2 = { .shard = 1, .offset = 796994512ULL },
            .wq = { .shard = 1, .offset = 806440400ULL },
            .bq = { .shard = 1, .offset = 806436304ULL },
            .wk = { .shard = 1, .offset = 796999120ULL },
            .bk = { .shard = 1, .offset = 796998608ULL },
            .wv = { .shard = 1, .offset = 814829520ULL },
            .bv = { .shard = 1, .offset = 814829008ULL },
            .wo = { .shard = 1, .offset = 798047696ULL },
            .wgate = { .shard = 1, .offset = 706816976ULL },
            .wup = { .shard = 1, .offset = 751905744ULL },
            .wdown = { .shard = 1, .offset = 661728208ULL }
        },
        // Camada 27
        {
            .norm1 = { .shard = 1, .offset = 815878096ULL },
            .norm2 = { .shard = 1, .offset = 951148496ULL },
            .wq = { .shard = 1, .offset = 960594384ULL },
            .bq = { .shard = 1, .offset = 960590288ULL },
            .wk = { .shard = 1, .offset = 951153104ULL },
            .bk = { .shard = 1, .offset = 951152592ULL },
            .wv = { .shard = 1, .offset = 968983504ULL },
            .bv = { .shard = 1, .offset = 968982992ULL },
            .wo = { .shard = 1, .offset = 952201680ULL },
            .wgate = { .shard = 1, .offset = 860970960ULL },
            .wup = { .shard = 1, .offset = 906059728ULL },
            .wdown = { .shard = 1, .offset = 815882192ULL }
        },
        // Camada 28
        {
            .norm1 = { .shard = 1, .offset = 970032080ULL },
            .norm2 = { .shard = 1, .offset = 1105302480ULL },
            .wq = { .shard = 1, .offset = 1114748368ULL },
            .bq = { .shard = 1, .offset = 1114744272ULL },
            .wk = { .shard = 1, .offset = 1105307088ULL },
            .bk = { .shard = 1, .offset = 1105306576ULL },
            .wv = { .shard = 1, .offset = 1123137488ULL },
            .bv = { .shard = 1, .offset = 1123136976ULL },
            .wo = { .shard = 1, .offset = 1106355664ULL },
            .wgate = { .shard = 1, .offset = 1015124944ULL },
            .wup = { .shard = 1, .offset = 1060213712ULL },
            .wdown = { .shard = 1, .offset = 970036176ULL }
        },
        // Camada 29
        {
            .norm1 = { .shard = 1, .offset = 1124186064ULL },
            .norm2 = { .shard = 1, .offset = 1259456464ULL },
            .wq = { .shard = 1, .offset = 1268902352ULL },
            .bq = { .shard = 1, .offset = 1268898256ULL },
            .wk = { .shard = 1, .offset = 1259461072ULL },
            .bk = { .shard = 1, .offset = 1259460560ULL },
            .wv = { .shard = 1, .offset = 1277291472ULL },
            .bv = { .shard = 1, .offset = 1277290960ULL },
            .wo = { .shard = 1, .offset = 1260509648ULL },
            .wgate = { .shard = 1, .offset = 1169278928ULL },
            .wup = { .shard = 1, .offset = 1214367696ULL },
            .wdown = { .shard = 1, .offset = 1124190160ULL }
        },
        // Camada 30
        {
            .norm1 = { .shard = 1, .offset = 1278340048ULL },
            .norm2 = { .shard = 1, .offset = 1413610448ULL },
            .wq = { .shard = 1, .offset = 1423056336ULL },
            .bq = { .shard = 1, .offset = 1423052240ULL },
            .wk = { .shard = 1, .offset = 1413615056ULL },
            .bk = { .shard = 1, .offset = 1413614544ULL },
            .wv = { .shard = 1, .offset = 1431445456ULL },
            .bv = { .shard = 1, .offset = 1431444944ULL },
            .wo = { .shard = 1, .offset = 1414663632ULL },
            .wgate = { .shard = 1, .offset = 1323432912ULL },
            .wup = { .shard = 1, .offset = 1368521680ULL },
            .wdown = { .shard = 1, .offset = 1278344144ULL }
        },
        // Camada 31
        {
            .norm1 = { .shard = 1, .offset = 1432494032ULL },
            .norm2 = { .shard = 1, .offset = 1567764432ULL },
            .wq = { .shard = 1, .offset = 1577210320ULL },
            .bq = { .shard = 1, .offset = 1577206224ULL },
            .wk = { .shard = 1, .offset = 1567769040ULL },
            .bk = { .shard = 1, .offset = 1567768528ULL },
            .wv = { .shard = 1, .offset = 1585599440ULL },
            .bv = { .shard = 1, .offset = 1585598928ULL },
            .wo = { .shard = 1, .offset = 1568817616ULL },
            .wgate = { .shard = 1, .offset = 1477586896ULL },
            .wup = { .shard = 1, .offset = 1522675664ULL },
            .wdown = { .shard = 1, .offset = 1432498128ULL }
        },
        // Camada 32
        {
            .norm1 = { .shard = 1, .offset = 1586648016ULL },
            .norm2 = { .shard = 1, .offset = 1721918416ULL },
            .wq = { .shard = 1, .offset = 1731364304ULL },
            .bq = { .shard = 1, .offset = 1731360208ULL },
            .wk = { .shard = 1, .offset = 1721923024ULL },
            .bk = { .shard = 1, .offset = 1721922512ULL },
            .wv = { .shard = 1, .offset = 1739753424ULL },
            .bv = { .shard = 1, .offset = 1739752912ULL },
            .wo = { .shard = 1, .offset = 1722971600ULL },
            .wgate = { .shard = 1, .offset = 1631740880ULL },
            .wup = { .shard = 1, .offset = 1676829648ULL },
            .wdown = { .shard = 1, .offset = 1586652112ULL }
        },
        // Camada 33
        {
            .norm1 = { .shard = 1, .offset = 1740802000ULL },
            .norm2 = { .shard = 1, .offset = 1876072400ULL },
            .wq = { .shard = 1, .offset = 1885518288ULL },
            .bq = { .shard = 1, .offset = 1885514192ULL },
            .wk = { .shard = 1, .offset = 1876077008ULL },
            .bk = { .shard = 1, .offset = 1876076496ULL },
            .wv = { .shard = 1, .offset = 1893907408ULL },
            .bv = { .shard = 1, .offset = 1893906896ULL },
            .wo = { .shard = 1, .offset = 1877125584ULL },
            .wgate = { .shard = 1, .offset = 1785894864ULL },
            .wup = { .shard = 1, .offset = 1830983632ULL },
            .wdown = { .shard = 1, .offset = 1740806096ULL }
        },
        // Camada 34
        {
            .norm1 = { .shard = 1, .offset = 1894955984ULL },
            .norm2 = { .shard = 1, .offset = 2030226384ULL },
            .wq = { .shard = 1, .offset = 2039672272ULL },
            .bq = { .shard = 1, .offset = 2039668176ULL },
            .wk = { .shard = 1, .offset = 2030230992ULL },
            .bk = { .shard = 1, .offset = 2030230480ULL },
            .wv = { .shard = 1, .offset = 2048061392ULL },
            .bv = { .shard = 1, .offset = 2048060880ULL },
            .wo = { .shard = 1, .offset = 2031279568ULL },
            .wgate = { .shard = 1, .offset = 1940048848ULL },
            .wup = { .shard = 1, .offset = 1985137616ULL },
            .wdown = { .shard = 1, .offset = 1894960080ULL }
        },
        // Camada 35
        {
            .norm1 = { .shard = 1, .offset = 2049109968ULL },
            .norm2 = { .shard = 1, .offset = 2184380368ULL },
            .wq = { .shard = 1, .offset = 2193826256ULL },
            .bq = { .shard = 1, .offset = 2193822160ULL },
            .wk = { .shard = 1, .offset = 2184384976ULL },
            .bk = { .shard = 1, .offset = 2184384464ULL },
            .wv = { .shard = 1, .offset = 2202215376ULL },
            .bv = { .shard = 1, .offset = 2202214864ULL },
            .wo = { .shard = 1, .offset = 2185433552ULL },
            .wgate = { .shard = 1, .offset = 2094202832ULL },
            .wup = { .shard = 1, .offset = 2139291600ULL },
            .wdown = { .shard = 1, .offset = 2049114064ULL }
        }
    }
};

#endif // QWEN_OFFSETS_3B_RAW_H
