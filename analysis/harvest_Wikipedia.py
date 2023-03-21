#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = " - "
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2021 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "


import wikipedia
wikipedia.set_lang('en')
import csv
import argparse
import pandas as pd
import numpy as np

argsparser = argparse.ArgumentParser()
args = argsparser.parse_args()


# topic list compiled by ChatGPT3
topics = ["Biodiversity", "Ecosystem", "Habitat", "Conservation", "Sustainable development", "Ecological succession",
          "Ecological footprint", "Ecological niche", "Ecological services", "Ecological functions", "Ecological restoration",
          "Ecological balance", "Ecological resilience", "Ecological diversity", "Ecological system", "Ecological interactions",
          "Ecological corridors", "Ecological monitoring", "Ecological stability", "Ecological risk assessment", "Ecological modelling",
          "Ecological genetics", "Ecological footprint", "DNA", "RNA", "proteins", "gene expression", "genetic engineering",
          "cells", "genetics", "evolution", "ecology", "biochemistry", "microbiology", "psychiatry", "biotechnology",
          "biophysics", "neuroscience", "cancer", "immunology", "epidemiology", "endocrinology", "virology", "botany",
          "zoology", "physiology", "pharmacology", "toxicology", "psychiatry", "biostatistics", "ecology", "biochemistry",
          "psychiatry", "psychiatry", "genetic engineering", "Cloning", "Recombinant DNA", "Bioprocessing", "Bioprinting",
          "Bioreactor", "Metabolic engineering", "Synthetic biology", "Biomanufacturing", "Immunotechnology",
          "Biopharmaceuticals", "Stem cells", "Tissue engineering", "Biomaterials", "Microbial biotechnology",
          "Biocatalysis", "Biopesticides", "Bioremediation", "Biofuels", "Bioseparation", "Biocomputing",
          "Biomedical engineering", "Biosensors", "Biomarkers", "Bioprospecting", "Bioplastics", "Biodegradable materials",
          "Biocatalysts", "Bioprocess control", "Bioprocess scale-up", "Bioprocess design", "Bioprocess modeling",
          "Bioprocess validation", "Bioprocess optimization", "Bioprocess monitoring", "Bioprocess safety",
          "Bioprocess regulations", "Bioprocess patenting", "Bioprocess commercialization", "Bioprocess investment",
          "Immune system", "Antigen", "Antibody", "T cells", "B cells", "Immunoglobulins", "MHC molecules", "Cytokines",
          "Lymphocytes", "Immunological memory", "Immune response", "Autoimmunity", "Allergy", "Immunodeficiency",
          "Immune tolerance", "Vaccines", "Immune surveillance", "Immune cell activation", "Immune cell differentiation",
          "Immune cell migration", "Immune cell signaling", "Immune cell apoptosis", "Immune cell senescence",
          "Immune cell regeneration", "Immune cell aging", "Immune cell metabolism", "Immune cell plasticity",
          "Immune cell interactions", "Immune cell metabolism", "Immune cell plasticity", "Immune cell interactions",
          "Immune cell education", "Immune cell activation", "Immune cell differentiation", "Immune cell migration",
          "Immune cell signaling", "Immune cell apoptosis", "Immune cell senescence", "Immune cell regeneration",
          "Immune cell aging", "Immune cell metabolism", "Immune cell plasticity", "Immune cell interactions",
          "Immune cell education", "Immune cell activation", "Immune cell differentiation", "Immune cell migration",
          "Immune cell signaling", "Immune cell apoptosis", "Immune cell senescence", "Mental health", "Depression",
          "Anxiety", "Psychotherapy", "Psychiatrist", "Medication", "Schizophrenia", "Bipolar disorder",
          "Personality disorder", "Post-traumatic stress disorder", "Addiction", "Eating disorder",
          "Obsessive-compulsive disorder", "Attention deficit hyperactivity disorder", "Autism spectrum disorder",
          "Dementia", "Mood disorder", "Substance abuse", "Trauma", "Stress", "Grief", "Sleep disorder",
          "Psychoanalysis", "Cognitive-behavioral therapy", "Dialectical behavior therapy", "Interpersonal therapy",
          "Family therapy", "Group therapy", "Art therapy", "Music therapy", "Animal assisted therapy", "Mindfulness",
          "Neuropsychiatry", "Forensic psychiatry", "Child psychiatry", "Geriatric psychiatry", "Clinical neuroscience",
          "Neuroimaging", "Neuropsychology", "Neuropharmacology", "Social psychiatry", "Community psychiatry",
          "Cultural psychiatry", "Global mental health", "Public mental health", "Mental health policy", "Mental health reform",
          "Mental health stigma", "Mental health advocacy", "Anatomy", "Physiology", "Pathology", "Microbiology",
          "Immunology", "Pharmacology", "Therapeutics", "Surgery", "Obstetrics", "Gynecology", "Pediatrics", "Geriatrics",
          "Oncology", "Cardiology", "Endocrinology", "Gastroenterology", "Hematology", "Infectious disease", "Nephrology",
          "Neurology", "Ophthalmology", "Orthopedics", "Otolaryngology", "Psychiatry", "Pulmonology", "Radiology",
          "Rheumatology", "Urology", "Anesthesiology", "Dermatology", "Emergency medicine", "Family medicine",
          "Internal medicine", "Physical medicine", "Sports medicine", "Clinical research", "Clinical trials",
          "Epidemiology", "Public health", "Global health", "Health policy", "Healthcare reform", "Medical ethics",
          "Medical education", "Medical technology", "Medical informatics", "Medical devices", "Telemedicine",
          "Medical imaging", "Medical robots", "Medical artificial intelligence", "Vitamins", "Minerals", "Herbs",
          "Probiotics", "Enzymes", "Amino acids", "Omega-3", "Antioxidants", "Fiber", "Phytochemicals", "Food-based supplements",
          "Herbal supplements", "Nutraceuticals", "Dietary supplements", "Functional food", "Food fortification",
          "Food enrichment", "Food ingredients", "Food additives", "Food labeling", "Food safety", "Food regulations",
          "Food standards", "Food science", "Food technology", "Food processing", "Food packaging", "Food preservation",
          "Food microbiology", "Food chemistry", "Food engineering", "Food biotechnology", "Food sustainability",
          "Food security", "Food waste", "Food traceability", "Food culture", "Food history", "Food anthropology",
          "Food trends", "Food marketing", "Food advertising", "Food literacy", "Food education", "Elder care",
          "Child care", "Disability care", "Mental health care", "Home care", "Residential care", "Hospice care",
          "Palliative care", "Long-term care", "End-of-life care", "Assisted living", "Nursing care", "Social care",
          "In-home care", "Community care", "Care coordination", "Care management", "Care planning", "Care assessment",
          "Care delivery", "Caregiving", "Care work", "Care work force", "Care work policies", "Care work rights",
          "Care work benefits", "Care work training", "Care work education", "Care work diversity", "Care work ethics",
          "Care work standards", "Care work regulations", "Care work reform", "Care work advocacy", "Care work research",
          "Care work innovation", "Care work technology", "Care work telecare", "Care work robots", "Care work artificial intelligence",
          "Plants", "Photosynthesis", "Taxonomy", "Ecology", "Phytochemistry", "Phylogenetics", "Genetics", "Evolution",
          "Morphology", "Anatomy", "Physiology", "Palynology", "Mycology", "Phycology", "Bryology", "Pteridology", "Systematics",
          "Dendrology", "Ethnobotany", "Phytopharmacology", "Plant breeding", "Plant biotechnology", "Plant pathology",
          "Plant physiology", "Plant ecology", "Plant physiology", "Plant physiology", "Plant physiology", "Plant psychiatry",
          "Plant physiology", "Plant psychiatry", "Plant physiology", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry",
          "Plant psychiatry", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry",
          "Plant psychiatry", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry", "Plant psychiatry",
          "Plant psychiatry", "Plant psychiatry", "Microorganisms", "Bacteria", "Viruses", "Fungi", "Protozoa", "Algae",
          "Microbial ecology", "Microbial diversity", "Microbial physiology", "Microbial genetics", "Microbial physiology",
          "Microbial physiology", "Microbial physiology", "Microbial physiology", "Microbial physiology", "Microbial psychiatry",
          "Microbial physiology", "Microbial psychiatry", "Microbial physiology", "Microbial psychiatry", "Microbial physiology",
          "Microbial psychiatry", "Microbial physiology", "Microbial psychiatry", "Nutrients", "Diet", "Metabolism",
          "Healthy eating", "Weight management", "Food groups", "Macronutrients", "Micronutrients", "Vitamins", "Minerals",
          "Amino acids", "Fatty acids", "Carbohydrates", "Proteins", "Fiber", "Water", "Energy balance", "Nutrition assessment",
          "Nutrition education", "Nutrition intervention", "Nutrition research", "Nutrition epidemiology", "Nutrition policy",
          "Nutrition regulations", "Nutrition standards", "Nutrition labeling", "Nutrition claims", "Nutrition and disease",
          "Nutrition and aging", "Nutrition and fitness", "Nutrition and pregnancy", "Nutrition and lactation",
          "Nutrition and childhood", "Nutrition and adolescence", "Nutrition and older adults", "Nutrition and public health",
          "Nutrition and sustainability", "Nutrition and food security", "Nutrition and global health", "Nutrition and food industry",
          "Nutrition and food technology", "Nutrition and food science", "Nutrition and food systems", "Crops", "Soils",
          "Fertilizers", "Pesticides", "Irrigation", "Genetics", "Plant breeding", "Agronomy", "Horticulture", "Entomology",
          "Pathology", "Weed science", "Soil science", "Water management", "Agricultural engineering", "Agricultural economics",
          "Agricultural extension", "Agricultural policy", "Agricultural sustainability", "Agricultural productivity",
          "Agricultural biotechnology", "Agricultural mechanization", "Agricultural waste management", "Agricultural conservation",
          "Agricultural ecology", "Agricultural meteorology", "Agricultural climatology", "Agricultural remote sensing",
          "Agricultural statistics", "Agricultural education", "Agricultural research", "Agricultural development",
          "Agricultural industry", "Agricultural market", "Agricultural trade", "Agricultural innovation", "Agricultural technology",
          "Agricultural robotics", "Agricultural artificial intelligence", "Agricultural precision", "Agricultural genomics",
          "SARS-CoV-2", "Coronavirus", "COVID-19", "Pandemic", "Virus", "Infection", "Symptoms", "Transmission", "Immunity",
          "Vaccines", "Treatment", "Mortality", "Masks", "Social distancing", "Quarantine", "Isolation", "Contact tracing",
          "Testing", "Lockdown", "Clinical trials", "Epidemiology", "Virology", "Immunology", "Public health", "Healthcare system",
          "Economic impact", "Mental health", "Remote work", "Education", "Travel restrictions", "Vaccine hesitancy",
          "Vaccine distribution", "Vaccine efficacy", "Vaccine safety", "Variants", "Mutation", "Antiviral drugs",
          "Therapeutic monoclonal antibodies", "Pneumonia", "ARDS", "Rehospitalization", "Mortality", "Health equity",
          "Global response", "Pandemic preparedness","Oncology", "Cancer cells", "Cancer biology", "Cancer genetics",
          "Cancer epidemiology", "Cancer screening", "Cancer diagnosis", "Cancer treatment", "Cancer surgery",
          "Cancer radiation therapy", "Cancer chemotherapy", "Cancer immunotherapy", "Cancer targeted therapy",
          "Cancer stem cells", "Cancer biomarkers", "Cancer prognosis", "Cancer survival", "Cancer prevention",
          "Cancer epidemiology", "Cancer genetics", "Cancer immunology", "Cancer pathology", "Cancer pharmacology",
          "Cancer statistics", "Cancer research", "Cancer clinical trials", "Cancer drug development", "Cancer nanotechnology",
          "Cancer informatics", "Cancer genomics", "Cancer epigenetics", "Cancer proteomics", "Cancer metabolomics",
          "Cancer systems biology", "Cancer precision medicine", "Cancer survivorship", "Cancer palliative care",
          "Cancer advocacy", "Anatomy", "Head and Neck", "Thorax", "Abdomen", "Pelvis", "Upper Limb", "Lower Limb",
          "Biochemistry", "Biostatistics", "Embryology", "Epidemiology", "Genetics", "Immunology", "Investigative Pathology",
          "Medical Statistics", "Medical Informatics", "Microbiology (includes Bacteriology)", "Parasitology", "Virology",
          "Neuroscience", "Cytology", "Toxicology", "Histology", "Histopathology", "Pathophysiology", "Pharmacology",
          "Psychiatry", "Bioethics", "Medicine", "Emergency Medicine", "General Practitioner", "Internal Medicine",
          "Cardiology", "Critical Care Medicine", "Endocrinology", "Gastroenterology", "Hepatology", "Geriatric Medicine",
          "Hematology", "Infectious Diseases", "AIDS Medicine", "Genito-Urinary Medicine", "Nephrology", "Oncology", "Primary Care",
          "Pulmonology", "Rheumatology", "Medical Genetics", "Neurology", "Occupational Medicine", "Anesthesiology", "Pain Medicine",
          "Dentistry", "Dental Public Health", "Endodontics", "Oral and Maxillofacial Pathology", "Oral and Maxillofacial Surgery",
          "Oral Medicine", "Oral Radiology", "Orthodontics", "Pediatric Dentistry", "Periodontics", "Prosthodontics",
          "Fixed Prosthodontics", "Removable Prosthodontics", "Dermatology", "Obstetrics & Gynecology", "Gynecology",
          "Gynecologic Surgery", "Gynecologic Oncology", "Maternal/Fetal Medicine", "Obstetrics", "Reproductive Endocrinology and Fertility",
          "Ophthalmology", "Cataract Surgery", "Oculoplastic Surgery", "Refractive Surgery", "Vitreo-Retinal Surgery",
          "Pathology", "Anatomic Pathology", "Autopsy", "Cytopathology", "Forensic Pathology", "Molecular Pathology",
          "Surgical Pathology", "Clinical Pathology", "Clinical Chemistry", "Cytology", "Hematopathology",
          "Medical Microbiology", "Transfusion Medicine", "Pediatrics", "Adolescent Medicine", "Child Abuse Medicine",
          "Developmental Pediatrics", "Neonatology", "Pediatric Cardiology", "Pediatric Critical Care", "Pediatric Endocrinology",
          "Physicians", "Physician Assistants", "Nurses", "Nurse Practitioners", "Nurse Specialists", "Nurse Anaesthetists",
          "Nurse Midwives", "Dietitians", "Midwives", "Therapists", "Physical Therapists", "Respiratory Therapists",
          "Occupational Therapists", "Speech Therapists", "Allied Health Care Professionals", "Independent Practitioners",
          "Dentists", "Podiatrists", "Audiologists", "Optometrists", "Dependent Practitioners", "Physical Therapists",
          "Occupational Therapists", "Diagnostic Procedures", "EMG", "Doppler", "Cardiology Diagnostic Procedures",
          "Cardiac Catheterization", "Echocardiogram", "Pericardiocentesis", "EKG", "12-lead", "24 Hour", "Stress Tests",
          "Stress Echo", "Stress EKG", "Pulmonary Diagnostic Procedures", "Lung Function Studies", "Thoracocentesis",
          "Pleural Biopsy", "Bronchoscopy", "Other", "Joint Aspiration", "Peritoneal Lavage", "Bone Marrow Aspiration",
          "Gastroenterological Diagnostic Procedures", "Colonoscopy +/- Biopsy", "Sigmoidoscopy +/- Biopsy",
          "Proctosigmoidoscopy", "Anoscopy", "ERCP", "MRCP", "Esophagogastroduodenoscopy (Upper Endoscopy)", "Endoscopic Ultrasound",
          "Capsule Enteroscopy", "Liver Biopsy", "Esophageal Manometry", "24 Hour pH Monitoring", "Gastric Analysis",
          "Pancreatic Function Tests", "Radiologic Diagnostic Procedures", "Ultrasound", "X-ray", "CT scan", "Spiral CT",
          "Contrast CT", "MRI Scan", "Nuclear Medicine", "Laboratory Procedures", "Blood Tests", "Blood Chemistry",
          "Serum Sodium", "Serum Potassium", "Serum Chloride", "Serum Bicarbonate", "Blood Urea Nitrogen", "Blood Glucose",
          "Serum Creatinine", "Cerebrospinal Fluid Tests", "Cultures", "Viral Culture", "Bacterial Culture", "Fungal Culture",
          "Serology", "Brain Tests", "PET", "EEG", "Dermatology", "Cutaneous conditions", "Acneiform eruptions",
          "Autoinflammatory syndromes", "Chronic blistering cutaneous conditions", "Conditions of the mucous membranes",
          "Conditions of the skin appendages", "Conditions of the subcutaneous fat", "Connective tissue diseases",
          "Abnormalities of dermal fibrous and elastic tissue", "Cutaneous congenital anomalies", "Dermal and subcutaneous growths",
          "Dermatitis", "Atopic dermatitis", "Contact dermatitis", "Eczeama", "Pustular dermatitis", "Seborrheic dermatitis",
          "Disturbances of human pigmentation", "Drug eruptions", "Endocrine-related cutaneous conditions",
          "Eosinophilic cutaneous conditions", "Epidermal nevi, neoplasms, cysts", "Erythemas", "Genodermatoses",
          "Infection-related cutaneous conditions", "Bacterium-related cutaneous conditions",
          "Mycobacterium-related cutaneous conditions", "Mycosis-related cutaneous conditions",
          "Parasitic infestations, stings, and bites of the skin", "Virus-related cutaneous conditions", "Lichenoid eruptions",
          "Lymphoid-related cutaneous conditions", "Melanocytic nevi and neoplasms", "Melanoma",
          "Monocyte- and macrophage-related cutaneous conditions", "Mucinoses", "Neurocutaneous conditions",
          "Noninfectious immunodeficiency-related cutaneous conditions", "Papulosquamous hyperkeratotic cutaneous conditions",
          "Palmoplantar keratodermas", "Pregnancy-related cutaneous conditions", "Pruritic skin conditions", "Psoriasis",
          "Reactive neutrophilic cutaneous conditions", "Recalcitrant palmoplantar eruptions", "Skin conditions resulting from errors in metabolism",
          "Skin conditions resulting from physical factors", "Ionizing radiation-induced cutaneous conditions",
          "Urticaria and angioedema", "Vascular-related cutaneous conditions", "Dermatologic drugs", "Anti-acne preparations",
          "Antibiotics and chemotherapeutics for dermatologic use", "Antifungals for dermatologic use", "Antipruritics",
          "Antipsoriatics", "Antiseptics and disinfectants", "Dermatologic preparations of corticosteroids",
          "Emollients and protectives", "Medicated dressings", "Preparations for treatment of wounds and ulcers",
          "Dermatologic procedures and surgery", "Dermatologic signs", "Dermatologic societies", "Dermatologic terminology",
          "Dermatologists", "Dermatology journals", "Hair", "Hair anatomy", "Hair psychiatry", "Nails", "Nail anatomy",
          "Nail psychiatry", "Skin", "Skin anatomy", "Skin psychiatry"]


print(len(topics))
topics = list(dict.fromkeys(topics))
print(len(topics))

#csvfile= csv.writer(open(args.output, 'a'))
#csvfile.writerow(['title', 'text'])

df = pd.DataFrame(columns=['category-id', 'text-id', 'text'])
for i in topics:
    try:
        item = wikipedia.page(i)

        row = pd.DataFrame({'category-id': 2, 'text-id':[item.title], 'text':[item.content]})

        df = pd.concat([df, row], ignore_index=True)
        df = df.replace(r'\n', ' ', regex=True)
        df['text'] = df['text'].str.replace('=', '')
        #df.dropna(inplace=True)


        #titel = item.title
        #content = item.content
        #content = content.strip()
        #infos = titel, content
        #print(df)
        #csvfile.writerow(infos)
    except Exception as e:
        print("Exception", e)

df.to_csv('data/popscience_wikipedia-text.csv', index=False)