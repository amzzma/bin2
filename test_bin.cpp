#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#define pln std::cout<<"\n";

/*
You should alter the structor definition and reading buffer codes below
if you don't read netT.bin as reference file.
*/
struct weight {
    char bp[15];
    char layer_1_head[21];
    std::unique_ptr<char[]> layer_1_data;
    char layer_1_end[10];
    char bXXX_head[18];
    std::unique_ptr<char[]> bXXX_data;
    char bXXX_end[7];
    char ep[8];
};

static void prc(char* cs, int32_t len)
{
    std::cout << std::string(cs, len) << "\n";
}


static int testbin2struct(weight &data) {
    int64_t l1dl = 1986204, l2dl = 3430716;
    int32_t bpl = 15, l1hl=21, l1el=10, l2hl=18, l2el=7, epl=8; 
    std::cout << std::fixed << std::setprecision(5);
    // weight data;

    std::ifstream inputFile("netT.bin", std::ios::binary);

    if (!inputFile) {
        std::cerr << "Error opening file!" << std::endl;
        exit(-1);
    }
    char* bp = data.bp, * layer_1_head = data.layer_1_head;
    data.layer_1_data = std::make_unique<char[]>(l1dl);
    char* layer_1_end = data.layer_1_end, * bXXX_head = data.bXXX_head;
    data.bXXX_data = std::make_unique<char[]>(l2dl);
    char* bXXX_end = data.bXXX_end, * ep = data.ep;

    inputFile.read(bp, static_cast<std::streamsize>(bpl) + l1hl);
    inputFile.read(reinterpret_cast<char*>(data.layer_1_data.get()), static_cast<std::streamsize>(l1dl));
    inputFile.read(layer_1_end, static_cast<std::streamsize>(l1el)+l2hl);
    inputFile.read(reinterpret_cast<char*>(data.bXXX_data.get()), static_cast<std::streamsize>(l2dl));
    inputFile.read(bXXX_end, static_cast<std::streamsize>(l2el) + epl);



    if (!inputFile) {
        std::cerr << "Error reading file!" << std::endl;
        exit(-1);
    }

    inputFile.close();

    //prc(data.bp, bpl);/*prc(data.layer_1_head, l1hl);*/
    //for(int i=0; i<l1dl/4 ; ++i)
    //{
    //    std::cout << reinterpret_cast<float*>(data.layer_1_data)[i] << " ";
    //}
    //pln
    //prc(data.layer_1_end, l2el);
    //prc(data.bXXX_head, l2hl);
    //for(int i=0; i<l2dl/4; ++i)
    //{
    //    std::cout << reinterpret_cast<float*>(data.bXXX_data)[i] << " ";
    //}
    //pln
    //prc(data.bXXX_end, l2el);prc(data.ep, epl);

    return 0;
}
