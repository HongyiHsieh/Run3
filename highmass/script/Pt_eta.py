# %%
import numpy as np
import json
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from pathlib import Path
from hist import Hist
import pandas as pd
import coffea
import argparse
import os
import shutil


# %%
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    def process(self, events):
        dataset = events.metadata["dataset"]

        results = {}
        results[dataset] = {
            "count": len(events)
        }
        #pT
        h_photon_EB_pt_leading = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EB_pt_leading", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )

        h_photon_EB_pt_subleading = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EB_pt_subleading", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )

        h_photon_EE_pt_leading = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EE_pt_leading", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )

        h_photon_EE_pt_subleading = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EE_pt_subleading", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )

        #eta
        h_photon_EB_eta_leading = (
            Hist.new
            .StrCat([], growth=True, name="dataset_photon_EB_eta_leading", label="dataset")
            .Reg(40, -2.5, 2.5, overflow=False, underflow=False, name="x", label = "eta")
            .Weight()
        )
        h_photon_EB_eta_subleading = (
            Hist.new
            .StrCat([], growth=True, name="dataset_photon_EB_eta_subleading", label="dataset")
            .Reg(40, -2.5, 2.5, overflow=False, underflow=False, name="x", label = "eta")
            .Weight()
        )
        h_photon_EE_eta_leading = (
            Hist.new
            .StrCat([], growth=True, name="dataset_photon_EE_eta_leading", label="dataset")
            .Reg(40, -2.5, 2.5, overflow=False, underflow=False, name="x", label = "eta")
            .Weight()
        )
        h_photon_EE_eta_subleading = (
            Hist.new
            .StrCat([], growth=True, name="dataset_photon_EE_eta_subleading", label="dataset")
            .Reg(40, -2.5, 2.5, overflow=False, underflow=False, name="x", label = "eta")
            .Weight()
        )

        photons = events.Photon
        photons["charge"] = ak.zeros_like(photons.pt)  # added this because 'charge'
        # events.Photon = events.Photon[events.Photon.electronVeto]
        # separate the events into EB and EE
        photon_EB = photons[photons.isScEtaEB]
        photon_EE = photons[photons.isScEtaEE]

        photon_EB_no_none = photon_EB[~ak.is_none(photon_EB, axis=1)]      
        photon_EE_no_none = photon_EE[~ak.is_none(photon_EE, axis=1)]      

        EB_photon_selection = photon_EB_no_none[ak.num(photon_EB_no_none,axis=1) > 1]
        EE_photon_selection = photon_EE_no_none[ak.num(photon_EE_no_none,axis=1) > 1]

        # EB_photon_selection_1 = ak.combinations(EB_photon_selection, 2, fields=["pho_lead", "pho_sublead"])
        # EE_photon_selection_1 = ak.combinations(EE_photon_selection, 2, fields=["pho_lead", "pho_sublead"])

        # EB_photon_selection_4mom = EB_photon_selection_1["pho_lead"] + EB_photon_selection_1["pho_sublead"]
        # EE_photon_selection_4mom = EE_photon_selection_1["pho_lead"] + EE_photon_selection_1["pho_sublead"]

        # EB_photon_selection_1 = EB_photon_selection_1[ak.argsort(EB_photon_selection_4mom.pt, ascending=False)]
        # EE_photon_selection_1 = EE_photon_selection_1[ak.argsort(EE_photon_selection_4mom.pt, ascending=False)]

        # EB_photon_selection = EB_photon_selection[ak.argsort(EB_photon_selection.pt, axis=1,ascending=False)]

        # print(len(EB_photon_selection_1.pho_lead.pt[:,0]))

        # EB_pt
        h_photon_EB_pt_leading.fill(dataset_photon_EB_pt_leading=dataset, x=EB_photon_selection.pt[:,0])
        h_photon_EB_pt_subleading.fill(dataset_photon_EB_pt_subleading=dataset, x=EB_photon_selection.pt[:,1])
        # EE_pt
        h_photon_EE_pt_leading.fill(dataset_photon_EE_pt_leading=dataset, x=EE_photon_selection.pt[:,0])
        h_photon_EE_pt_subleading.fill(dataset_photon_EE_pt_subleading=dataset, x=EE_photon_selection.pt[:,1])

        results["photon_EB_pt_leading"] = h_photon_EB_pt_leading
        results["photon_EB_pt_subleading"] = h_photon_EB_pt_subleading
        results["photon_EE_pt_leading"] = h_photon_EE_pt_leading
        results["photon_EE_pt_subleading"] = h_photon_EE_pt_subleading

        #EB_eta
        h_photon_EB_eta_leading.fill(dataset_photon_EB_eta_leading=dataset, x=EB_photon_selection.eta[:,0])
        h_photon_EB_eta_subleading.fill(dataset_photon_EB_eta_subleading=dataset, x=EB_photon_selection.eta[:,1])
        #EE_eta
        h_photon_EE_eta_leading.fill(dataset_photon_EE_eta_leading=dataset, x=EE_photon_selection.eta[:,0])
        h_photon_EE_eta_subleading.fill(dataset_photon_EE_eta_subleading=dataset, x=EE_photon_selection.eta[:,1])

        results["photon_EB_eta_leading"] = h_photon_EB_eta_leading
        results["photon_EB_eta_subleading"] = h_photon_EB_eta_subleading
        results["photon_EE_eta_leading"] = h_photon_EE_eta_leading
        results["photon_EE_eta_subleading"] = h_photon_EE_eta_subleading

        return results
    
    def postprocess(self, accumulator):
        pass


# %%
for f in os.listdir("."):
    if f.endswith('.json'):
        with open(os.path.join(".", f)) as file:
            sample_dict = json.load(file)

# %%
run = processor.Runner(
    # executor=processor.IterativeExecutor(),
    executor=processor.FuturesExecutor(workers=10), # user 4 cores
    schema=NanoAODSchema
)

results = run(
    sample_dict,
    treename="Events",
    processor_instance=MyProcessor(),
)

# %%
# cell 23
shutil.rmtree('./parquet', ignore_errors=True)

plots_dir = './parquet'
Path(plots_dir).mkdir(exist_ok=True)
for key in sample_dict :
    #pT
    value_EB_pt_leading = results[f"photon_EB_pt_leading"][{f"dataset_photon_EB_pt_leading":key}].values()
    value_EB_pt_subleading = results[f"photon_EB_pt_subleading"][{f"dataset_photon_EB_pt_subleading":key}].values()
    value_EE_pt_leading = results[f"photon_EE_pt_leading"][{f"dataset_photon_EE_pt_leading":key}].values()
    value_EE_pt_subleading = results[f"photon_EE_pt_subleading"][{f"dataset_photon_EE_pt_subleading":key}].values()
    #eta
    value_EB_eta_leading = results[f"photon_EB_eta_leading"][{f"dataset_photon_EB_eta_leading":key}].values()
    value_EB_eta_subleading = results[f"photon_EB_eta_subleading"][{f"dataset_photon_EB_eta_subleading":key}].values()
    value_EE_eta_leading = results[f"photon_EE_eta_leading"][{f"dataset_photon_EE_eta_leading":key}].values()
    value_EE_eta_subleading = results[f"photon_EE_eta_subleading"][{f"dataset_photon_EE_eta_subleading":key}].values()

    parquet_data={
            'pt_EB_leading':value_EB_pt_leading,
            'pt_EB_subleading':value_EB_pt_subleading,
            'pt_EE_leading':value_EE_pt_leading,
            'pt_EE_subleading':value_EE_pt_subleading,
            'eta_EB_leading':value_EB_eta_leading,
            'eta_EB_subleading':value_EB_eta_subleading,
            'eta_EE_leading':value_EE_eta_leading,
            'eta_EE_subleading':value_EE_eta_subleading
    }

    df_pt = pd.DataFrame(data=parquet_data)
    df_pt.to_parquet(f'./parquet/{key}_photon.parquet')