{
  // Open the two input files
  // TFile* f1 = new TFile("../../../../data/20210305_20230731_new_ihep.root", "UPDATE");
  TFile* f1 = new TFile("../../../../data/KM2A_all_final2.root", "UPDATE");
  
  if(!f1 || f1->IsZombie()){
      cout<<"Unable to open the input file"<<endl;
      return -1;
      }


//   // TFile* outputFile = TFile::Open("./output.root", "RECREATE");
//   //   if(!outputFile || outputFile->IsZombie()){
//   //         cout<<"Unable to create the output file"<<endl;
//   //         return -1;
//   //   }
//   // Loop over all the trees in the first file


  const int bins=14;


  int i = 0;
  TDirectoryFile* outdir[bins];
  TDirectoryFile* outdir2[bins];  
  TTree* tree1[bins];
  TTree* tree2[bins];
  TTree* merge[bins];
  TTree* mergebkg[bins];
  TTree* merge2[bins];
  TTree* mergebkg2[bins];
  double count[bins], countbkg[bins],newvar[bins],newvarbkg[bins], newvar2[bins], newvarbkg2[bins];
  while (i<=bins-1) {
      if (i+bins<=9){
        outdir[i] = new TDirectoryFile(Form("nHit0%d",i+bins), "Merged Trees");
      }else{
        outdir[i] = new TDirectoryFile(Form("nHit%d",i+bins), "Merged Trees2");
      }

      if (i+bins+bins<=9){
        outdir2[i] = new TDirectoryFile(Form("nHit0%d",i+bins+bins), "Merged Trees");
      }else{
        outdir2[i] = new TDirectoryFile(Form("nHit%d",i+bins+bins), "Merged Trees2"); 
      }
      
      if (i<=9){
        tree1[i] = (TTree*)f1->Get(Form("nHit0%d/data;1",i));
      }else{
        tree1[i] = (TTree*)f1->Get(Form("nHit%d/data;1",i));
      }
      tree1[i]->SetBranchAddress("count", &count[i]);

      if(!tree1[i]){
          cout<<"Unable to find the input tree"<<endl;
          return -1;
      }

      if (i<=9){
        tree2[i] = (TTree*)f1->Get(Form("nHit0%d/bkg;1",i));
      }else{
        tree2[i] = (TTree*)f1->Get(Form("nHit%d/bkg;1",i));
      }
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
      merge2[i] = new TTree("data", "data Tree");
      merge2[i]->Branch("count", &newvar2[i]);

      // mergebkg[i] = tree2[i]->CloneTree();
      mergebkg[i] = new TTree("bkg", "data Tree");
      mergebkg[i]->Branch("count", &newvarbkg[i]);
      mergebkg2[i] = new TTree("bkg", "data Tree");
      mergebkg2[i]->Branch("count", &newvarbkg2[i]);
     i++;
  }


  for (int i = bins-1;i>=0;i--){
    for (Long64_t j = 0; j < tree1[i]->GetEntries(); j++) {
        merge[i]->GetEntry(j);
        mergebkg[i]->GetEntry(j);
        merge2[i]->GetEntry(j);
        mergebkg2[i]->GetEntry(j);
        newvar[i]=0;
        newvarbkg[i]=0;
        newvar2[i]=0;
        newvarbkg2[i]=0;
        for (int k = i;k<=bins-1;k++){
          tree1[k]->GetEntry(j);
          tree2[k]->GetEntry(j);
          if(j==1000000){
            cout<<i<<","<<k<<","<<newvar[i]<<","<<count[k]<<endl;
          }
          newvar[i] += count[k];
          newvarbkg[i] += countbkg[k];
        }
        for (int k = 0;k<=i;k++){
          tree1[k]->GetEntry(j);
          tree2[k]->GetEntry(j);
          if(j==1000000){
            cout<<i<<","<<k<<","<<newvar[i]<<","<<count[k]<<endl;
          }
          newvar2[i] += count[k];
          newvarbkg2[i] += countbkg[k];
        }
        merge[i]->Fill(); // 将相加后的结果填充到新的TTree中
        mergebkg[i]->Fill(); // 将相加后的结果填充到新的TTree中
        merge2[i]->Fill(); // 将相加后的结果填充到新的TTree中
        mergebkg2[i]->Fill(); // 将相加后的结果填充到新的TTree中
    }
    outdir[i]->WriteTObject(merge[i],"data");
    outdir[i]->WriteTObject(mergebkg[i],"bkg");
    outdir2[i]->WriteTObject(merge2[i],"data");
    outdir2[i]->WriteTObject(mergebkg2[i],"bkg");
  }

  // outputFile->Close();
  f1->Close();
}
