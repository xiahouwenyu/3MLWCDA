{
  // Open the two input files
  TFile* f1 = new TFile("../../../../data/20210305_20230731_new_ihep.root", "UPDATE");
  if(!f1 || f1->IsZombie()){
      cout<<"Unable to open the input file"<<endl;
      return -1;
      }


  // TFile* outputFile = TFile::Open("./output.root", "RECREATE");
  //   if(!outputFile || outputFile->IsZombie()){
  //         cout<<"Unable to create the output file"<<endl;
  //         return -1;
  //   }
  // Loop over all the trees in the first file
  int i = 0;
  TDirectoryFile* outdir[6]; 
  TTree* tree1[6];
  TTree* tree2[6];
  TTree* merge[6];
  TTree* mergebkg[6];
  double count[6], countbkg[6],newvar[6],newvarbkg[6];
  while (i<=5) {
      if (i+6<=9){
        outdir[i] = new TDirectoryFile(Form("nHit0%d",i+6), "Merged Trees");
      }else{
        outdir[i] = new TDirectoryFile(Form("nHit%d",i+6), "Merged Trees");
      }
      

      tree1[i] = (TTree*)f1->Get(Form("nHit0%d/data;1",i));
      tree1[i]->SetBranchAddress("count", &count[i]);

      if(!tree1[i]){
          cout<<"Unable to find the input tree"<<endl;
          return -1;
      }

      tree2[i] = (TTree*)f1->Get(Form("nHit0%d/bkg;1",i));
      tree2[i]->SetBranchAddress("count", &countbkg[i]);
      if(!tree2[i]){
          cout<<"Unable to find the input tree"<<endl;
          return -1;
      }
      cout<<tree1[i]->GetName()<<endl;
      cout<<tree2[i]->GetName()<<endl;
      // merge[i] = tree1[i]->CloneTree();
      merge[i] = new TTree("data", "data Tree");
      merge[i]->Branch("count", &newvar[i]);

      // mergebkg[i] = tree2[i]->CloneTree();
      mergebkg[i] = new TTree("bkg", "data Tree");
      mergebkg[i]->Branch("count", &newvarbkg[i]);
     i++;
  }


for (int i = 5;i>=0;i--){
    for (Long64_t j = 0; j < tree1[i]->GetEntries(); j++) {
        merge[i]->GetEntry(j);
        mergebkg[i]->GetEntry(j);
        newvar[i]=0;
        newvarbkg[i]=0;
        for (int k = i;k<=5;k++){
          tree1[k]->GetEntry(j);
          tree2[k]->GetEntry(j);
          if(j==1000000){
            cout<<i<<","<<k<<","<<newvar[i]<<","<<count[k]<<endl;
          }
          newvar[i] += count[k];
          newvarbkg[i] += countbkg[k];
        }
        merge[i]->Fill(); // 将相加后的结果填充到新的TTree中
        mergebkg[i]->Fill(); // 将相加后的结果填充到新的TTree中
    }
    outdir[i]->WriteTObject(merge[i],"data");
    outdir[i]->WriteTObject(mergebkg[i],"bkg");
}

  // outputFile->Close();
  f1->Close();
}
