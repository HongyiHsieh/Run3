from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from higgs_dna.utils.misc_utils import choose_jet

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections

import functools
import operator
import os
import warnings
from typing import Any, Dict, List, Optional
import awkward
import numpy
import sys
import vector
from coffea import processor
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class HighMassProcessor(HggBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group=".*DoubleEG.*",
        analysis="mainAnalysis",
        skipCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet"
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            output_location=output_location,
            taggers=taggers,
            trigger_group=trigger_group,
            analysis=analysis,
            skipCQR=skipCQR,
            skipJetVetoMap=skipJetVetoMap,
            year=year,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            output_format=output_format
        )


    def choose_nth_object_variable(self, variable, n, fill_value):
        """
        this helper function is used to create flat collection from a jagged collection,
        parameters:
        * variable: (awkward array) selected variable from the object
        * n: (int) nth object to be selected
        * fill_value: (float) value with which to fill the padded none.
        """
        variable = variable[
            awkward.local_index(variable) == n
        ]
        variable = awkward.pad_none(
            variable, 1
        )
        variable = awkward.flatten(
            awkward.fill_none(variable, fill_value)
        )
        return variable

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                awkward.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(awkward.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(awkward.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                awkward.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply jetvetomap: only retain events that without any jets in the EE leakage region
        # if not self.skipJetVetoMap:
        #     events = jetvetomap(
        #         events, logger, dataset_name, year=self.year[dataset_name][0]
        #     )
        # metadata array to append to higgsdna output
        metadata = {}

        # if self.data_kind == "mc":
        #     # Add sum of gen weights before selection for normalisation in postprocessing
        #     metadata["sum_genw_presel"] = str(awkward.sum(events.genWeight))
        # else:
        #     metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        # events = self.apply_filters_and_triggers(events)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        # events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # add veto EE leak branch for photons, could also be used for electrons
        # if (
        #     self.year[dataset_name][0] == "2022EE"
        #     or self.year[dataset_name][0] == "2022postEE"
        # ):
        #     events.Photon = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        # for correction_name in correction_names:
        #     if correction_name in available_object_corrections.keys():
        #         logger.info(
        #             f"Applying correction {correction_name} to dataset {dataset_name}"
        #         )
        #         varying_function = available_object_corrections[correction_name]
        #         events = varying_function(
        #             events=events, year=self.year[dataset_name][0]
        #         )
        #     elif correction_name in available_weight_corrections:
        #         # event weight corrections will be applied after photon preselection / application of further taggers
        #         continue
        #     else:
        #         # may want to throw an error instead, needs to be discussed
        #         warnings.warn(f"Could not process correction {correction_name}.")
        #         continue

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # systematic object variations
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                if systematic_dct["object"] == "Photon":
                    logger.info(
                        f"Adding systematic {systematic_name} to photons collection of dataset {dataset_name}"
                    )
                    original_photons.add_systematic(
                        # passing the arguments here explicitly since I want to pass the events to the varying function. If there is a more elegant / flexible way, just change it!
                        name=systematic_name,
                        kind=systematic_dct["args"]["kind"],
                        what=systematic_dct["args"]["what"],
                        varying_function=functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events=events,
                            year=self.year[dataset_name][0],
                        )
                        # name=systematic_name, **systematic_dct["args"]
                    )
                # to be implemented for other objects here
            elif systematic_name in available_weight_systematics:
                # event weight systematics will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(
                    f"Could not process systematic variation {systematic_name}."
                )
                continue

        # Applying systematic variations
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            # if self.chained_quantile is not None:
            #     photons = self.chained_quantile.apply(photons, events)
            # # recompute photonid_mva on the fly
            # if self.photonid_mva_EB and self.photonid_mva_EE:
            #     photons = self.add_photonid_mva(photons, events)

            # photon preselection
            # photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])

            # sort photons in each event descending in pt
            # make descending-pt combinations of photons
            photons = photons[awkward.argsort(photons.pt, ascending=False)]
            photons["charge"] = awkward.zeros_like(
                photons.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            # photon_EB = photons[photons.isScEtaEB[:,:2] & photons.isScEtaEE[:,:2]]
            # photon_EE = photons[photons.isScEtaEE]

            # photon_EB_no_none = photon_EB[~awkward.is_none(photon_EB, axis=1)]      
            # photon_EE_no_none = photon_EE[~awkward.is_none(photon_EE, axis=1)] 

            # EB_photon_selection = photon_EB_no_none[awkward.num(photon_EB_no_none,axis=1) > 1]
            # EE_photon_selection = photon_EE_no_none[awkward.num(photon_EE_no_none,axis=1) > 1]


            diphotons = awkward.combinations(
                photons, 2, fields=["pho_lead", "pho_sublead"]
            )

            # the remaining cut is to select the leading photons
            # the previous sort assures the order
            # diphotons = diphotons[
            #     diphotons["pho_lead"].pt > self.min_pt_lead_photon
            # ]

            # now turn the diphotons into candidates with four momenta and such
            diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
            diphotons["pt"] = diphoton_4mom.pt
            diphotons["eta"] = diphoton_4mom.eta
            diphotons["phi"] = diphoton_4mom.phi
            diphotons["mass"] = diphoton_4mom.mass
            diphotons["charge"] = diphoton_4mom.charge

            # diphotons["EB_pt"] = EB_photon_selection.pt
            # diphotons["EE_pt"] = EE_photon_selection.pt
            
            diphoton_pz = diphoton_4mom.z
            diphoton_e = diphoton_4mom.energy

            diphotons["rapidity"] = 0.5 * numpy.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))

            diphotons = awkward.with_name(diphotons, "PtEtaPhiMCandidate")

            # sort diphotons by pT
            # diphotons = diphotons[
            #     awkward.argsort(diphotons.pt, ascending=False)
            # ]

            # # Determine if event passes fiducial Hgg cuts at detector-level
            # if self.fiducialCuts == 'classical':
            #     fid_det_passed = (diphotons.pho_lead.pt / diphotons.mass > 1 / 3) &\
            #                      (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4) & \
            #                      (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10) & \
            #                      ((diphotons.pho_sublead.pfRelIso03_all_quadratic * diphotons.pho_sublead.pt) < 10) & \
            #                      (numpy.abs(diphotons.pho_lead.eta) < 2.5) & \
            #                      (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
            # elif self.fiducialCuts == 'geometric':
            #     fid_det_passed = (numpy.sqrt(diphotons.pho_lead.pt * diphotons.pho_sublead.pt) / diphotons.mass > 1 / 3) & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4) & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10) & (diphotons.pho_sublead.pfRelIso03_all_quadratic * diphotons.pho_sublead.pt < 10) & (numpy.abs(diphotons.pho_lead.eta) < 2.5) & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
            # elif self.fiducialCuts == 'none':
            #     fid_det_passed = diphotons.pho_lead.pt > -10  # This is a very dummy way but I do not know how to make a true array of outer shape of diphotons
            # else:
            #     warnings.warn("You chose %s the fiducialCuts mode, but this is currently not supported. You should check your settings. For this run, no fiducial selection at detector level is applied." % self.fiducialCuts)
            #     fid_det_passed = diphotons.pho_lead.pt > -10

            # diphotons = diphotons[fid_det_passed]

            # if self.data_kind == "mc":
            #     # Add the fiducial flags for particle level
            #     diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
            #     diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

            #     diphotons['PTH'], diphotons['YH'], diphotons['PHIH'] = get_higgs_gen_attributes(events)

            #     genJets = get_genJets(self, events, pt_cut=30., eta_cut=2.5)
            #     diphotons['NJ'] = awkward.num(genJets)
            #     diphotons['PTJ0'] = choose_jet(genJets.pt, 0, -999.0)  # Choose zero (leading) jet and pad with -999 if none

            # baseline modifications to diphotons
            # if self.diphoton_mva is not None:
            #     diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            # jet_variables
            jets = awkward.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": awkward.zeros_like(
                        jets.pt
                    ),  # added this because jet charge is not a property of photons in nanoAOD v11. We just need the charge to build jet collection.
                    "hFlav": jets.hadronFlavour
                    if self.data_kind == "mc"
                    else awkward.zeros_like(jets.pt),
                    "btagDeepFlav_B": jets.btagDeepFlavB,
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "jetId": jets.jetId,
                }
            )
            jets = awkward.with_name(jets, "PtEtaPhiMCandidate")

            electrons = awkward.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                }
            )
            electrons = awkward.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

            muons = awkward.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.charge,
                    "tightId": events.Muon.tightId,
                    "mediumId": events.Muon.mediumId,
                    "looseId": events.Muon.looseId,
                    "isGlobal": events.Muon.isGlobal,
                }
            )
            muons = awkward.with_name(muons, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]
            jets = jets[awkward.argsort(jets.pt, ascending=False)]

            events["sel_jets"] = jets
            n_jets = awkward.num(jets)
            Njets2p5 = awkward.num(jets[(jets.pt > 30) & (numpy.abs(jets.eta) < 2.5)])

            first_jet_pt = choose_jet(jets.pt, 0, -999.0)
            first_jet_eta = choose_jet(jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(jets.mass, 0, -999.0)
            first_jet_charge = choose_jet(jets.charge, 0, -999.0)

            second_jet_pt = choose_jet(jets.pt, 1, -999.0)
            second_jet_eta = choose_jet(jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(jets.mass, 1, -999.0)
            second_jet_charge = choose_jet(jets.charge, 1, -999.0)

            diphotons["first_jet_pt"] = first_jet_pt
            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass
            diphotons["first_jet_charge"] = first_jet_charge

            diphotons["second_jet_pt"] = second_jet_pt
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass
            diphotons["second_jet_charge"] = second_jet_charge

            diphotons["n_jets"] = n_jets
            diphotons["Njets2p5"] = Njets2p5

            #### start of your code ###
            
            # make two combinations of jets
            dijets = awkward.combinations(
               jets, 2, fields=("first_jet", "second_jet")
            )
            
            # now turn the dijets into candidates with four momenta and such
            dijets_4mom = dijets["first_jet"] + dijets["second_jet"]
            dijets["pt"] = dijets_4mom.pt
            dijets["eta"] = dijets_4mom.eta
            dijets["phi"] = dijets_4mom.phi
            dijets["mass"] = dijets_4mom.mass
            dijets["charge"] = dijets_4mom.charge
            dijets = awkward.with_name(dijets, "PtEtaPhiMCandidate")
            
            # add delta eta and delta phi of the dijet system 
            dijets["delta_eta"] = abs(dijets["first_jet"].eta - dijets["second_jet"].eta)
            dijets["delta_phi"] = dijets.first_jet.delta_phi(dijets.second_jet) # PtEtaPhiMCandidate already has a method from which you can calculate delta phi
            
            # choose the leading dijet combination to save
            # note the jet collection was already already ordered according to pT in line 415
            # so selecting the first dijet combination will automatically select the diject combination formed by the two leading-pT jets
            dijet_pt = self.choose_nth_object_variable(dijets.pt, 0, -999.0)
            dijet_eta = self.choose_nth_object_variable(dijets.eta, 0, -999.0)
            dijet_phi = self.choose_nth_object_variable(dijets.phi, 0, -999.0)
            dijet_mass = self.choose_nth_object_variable(dijets.mass, 0, -999.0)
            dijet_charge = self.choose_nth_object_variable(dijets.charge, 0, -999.0)
            dijet_delta_eta = self.choose_nth_object_variable(dijets.delta_eta, 0, -999.0)
            
            # as dijet is a PtEtaPhiMCandidate, calling dijets.delta_phi would call the method instead of the delta phi values. So use dijets["delta_phi"]
            dijet_delta_phi = self.choose_nth_object_variable(dijets["delta_phi"], 0, -999.0)
            
            diphotons["dijet_pt"] = dijet_pt
            diphotons["dijet_eta"] = dijet_eta
            diphotons["dijet_phi"] = dijet_phi
            diphotons["dijet_mass"] = dijet_mass
            diphotons["dijet_charge"] = dijet_charge
            diphotons["dijet_delta_eta"] = dijet_delta_eta
            diphotons["dijet_delta_phi"] = dijet_delta_phi

            ### end of your code ###

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            diphotons[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted]

            # diphotons = awkward.firsts(diphotons)
            # set diphotons as part of the event record
            events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(diphotons)
                    | awkward.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~awkward.is_none(diphotons)
                diphotons = diphotons[selection_mask]

            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"diphotons_{do_variation}"][
                                selection_mask
                            ],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"diphotons_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask]
                    * diphotons["weight_central"]
                )
                diphotons["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = awkward.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = awkward.ones_like(diphotons["event"])

            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    akarr = diphoton_ak_array(self, diphotons)

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = (
                    events.behavior[
                        "__events_factory__"
                    ]._partition_key.replace("/", "_")
                    + ".%s" % self.output_format
                )
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(do_variation)
                if self.output_format == "root":
                    dump_pandas(self, df, fname, self.output_location, subdirs)
                else:
                    dump_ak_array(
                        self, akarr, fname, self.output_location, metadata, subdirs,
                    )

        return histos_etc

    def process_extra(self, events: awkward.Array) -> awkward.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass