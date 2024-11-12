#include <TFile.h>
#include <TTree.h>
#include <TParameter.h>

int main(int argc, char *argv[])
{
    TFile *file = TFile::Open(argv[1], "UPDATE");
    // TTree *BinInfo = (TTree*)file->Get("BinInfo");
    // file->Delete("BinInfo");
    // try {
    //     // TTree* newTree = BinInfo->CopyTree(Form("name>=%i&&name<=%i",std::stoi(argv[2]),std::stoi(argv[3])));
    //     TTree* newTree = BinInfo->CopyTree(Form("name>=%s&&name<=%s",argv[2],argv[3]));
    //     // TTree* newTree = BinInfo->CopyTree(Form("name >= %i && name <= %i", std::stoi(argv[2]), std::stoi(argv[3])));
    // }catch (const std::exception& e) {
    //     // std::cout << "BinInfo name data type: " << BinInfo->GetBranch("name")->GetLeaf("name")->GetTypeName() << std::endl;
    //     // std::cerr << "BinInfo name value: " << BinInfo->GetBranch("name")->GetLeaf("name")->GetValue() << std::endl;
    //     std::cerr << "argv[2] data type: " << typeid(argv[2]).name() << ", value: " << argv[2] << std::endl;
    //     std::cerr << "argv[3] data type: " << typeid(argv[3]).name() << ", value: " << argv[3] << std::endl;
    //     std::cerr << "Exception caught: " << e.what() << std::endl;
    //     TTree* newTree = BinInfo->CopyTree(Form("name>=%i&&name<=%i",std::stoi(argv[2]),std::stoi(argv[3])));
    // }
    // TTree* newTree = BinInfo->CopyTree(Form("name>=%s&&name<=%s",argv[2],argv[3]));
    // TTree* newTree = BinInfo->CopyTree(Form("name>=%i&&name<=%i",std::stoi(argv[2]),std::stoi(argv[3])));
    for (int i = 0; i <14;i++) {
        TTree *data = (TTree*)file->Get(Form("nHit%02d/data",i));
        TTree *bkg = (TTree*)file->Get(Form("nHit%02d/bkg",i));
        if (data && bkg) {
            try {
                TParameter<int> *obj1 = new TParameter<int>("Nside",1024);
                TParameter<int> *obj2 = new TParameter<int>("Scheme",0);
                
                data->GetUserInfo()->Add(obj1);
                data->GetUserInfo()->Add(obj2);
                bkg->GetUserInfo()->Add(obj1);
                bkg->GetUserInfo()->Add(obj2);
            }catch (const std::exception& e) {
                std::cerr << "Exception caught: " << e.what() << std::endl;
            }
        }
    }
    // BinInfo->Write();
    file->Write();
    file->Close();
}