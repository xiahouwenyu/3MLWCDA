#include <TFile.h>
#include <TTree.h>
#include <TParameter.h>

void Add_UserInfo(char filename[]){
    // TFile *file = TFile::Open("../data/residual_all.root", "UPDATE");
    TFile *file = TFile::Open(filename, "UPDATE");

    for (int i = 0; i <6;i++) {
        TTree *data = (TTree*)file->Get(Form("nHit%02d/data",i));
        TTree *bkg = (TTree*)file->Get(Form("nHit%02d/bkg",i));

        TParameter<int> *obj1 = new TParameter<int>("Nside",1024);
        TParameter<int> *obj2 = new TParameter<int>("Scheme",0);
        
        data->GetUserInfo()->Add(obj1);
        data->GetUserInfo()->Add(obj2);
        bkg->GetUserInfo()->Add(obj1);
        bkg->GetUserInfo()->Add(obj2);
    }
    file->Write();
    file->Close();
}