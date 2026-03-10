from dataclasses import dataclass, asdict, replace, field
import json, pathlib
from typing import Optional, List, Dict

@dataclass(frozen=True)
class HyperParams:
    #---------------USER INPUT---------------#
    #-----SAVE LOGISTICS-----#
    #where the results get saved
    OUT_DIR: Optional[str] = None   
    #where the data folder is located 
    DATA_PATH:  Optional[str] = None   
    #where the summary of results gets saved
    RESULT_CSV:  Optional[str] = None   
    #tag for above summary file 
    RUN_NAME: Optional[str] = None   
    #options: AI-READI-OG, AI-READI-FULL, MIMIC
    DATASET:  Optional[str] = None   

    #-----TRAINING-----# 
    SEED: int = 0 
    ITERS: int = 25_000

    #------------AUTOMATICALLY CONFIGURED------------#
    #-----DATA LOGISTICS-----#
    #where min max file gets saved
    NPY_PATH:  Optional[str] = None   
    #list of numeric columns (IN ORDER)
    

    #-----MODEL PARAMETERS - THESE GET OVERWRITTEN IN SEARCH-----#
    DEVICE: str = 'cpu'
    BATCH: int = 256 
    NOISE_DIM: int = 32  
    GRADIENT_PENALTY: int = 5
    G_LR: float = 0.0001 
    G_H: int = 64 
    D_LR: float = 1e-05
    D_H: int = 128 
    DISC_STEPS: int = 5 
    NUM_SAMPLES: int = 5_000
    G_NUM_LAYERS: int = 2
    D_NUM_LAYERS: int = 2
    
    #PPO SPECIFIC  
    USE_TANH: bool = True  
    #number of times to iterate over PPO training loop (gen steps)
    PPO_EPOCHS: int = 3 
    MEAN_PENALTY_SCALE: float = 0.2 
    VF_COEF: float = 0.5 #default in PPO, pretty standard
    CLIP_EPS: float = 0.1 #can also try 0.2 
    ENT_BETA: float = 1e-3 #can also try 0.01 
    #simple clip on tanh to avoid bugs 
    EPS: float = 1e-6 

    # Fraction of the training data to use
    FRAC: float = 1.0

    # Place to store other params generated during training
    STATE: Dict = field(default_factory=dict)  
    RESUME_ITER: int = 0
    PENALIZED: bool = False
    REGRESSION_LAMBDA: float = 1.0
    ANNEAL_START: float = 0.0

    #
    EVAL_ONLY: bool = False


    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=2))

    def override(self, **kwargs):
        return replace(self, **kwargs)

    def load(self, path):
        with open(path, 'r') as file:
            loaded_params = json.load(file)
        
        return replace(self, **loaded_params)



@dataclass(frozen=True)
class HyperParams_ACS(HyperParams):
  
    #list of numeric columns (IN ORDER)
    NUM_COLS: List[str] = field(default_factory=lambda: ['Age', 'Years of School', 'Public Assistance Income'])

    #list of categorical columns (IN ORDER)
    BIN_COLS: List[str] = field(default_factory=lambda:['Male', 'Disability',
                                                        '1yr Mobility', 'Native', 'Institutionalized',
                                                        'Hearing Difficulty', 'Vision Difficulty', 'Cognitive Difficulty', 
                                                        'Hispanic/Latino', 'Health Insurance', "Public Assistance Binary"
                                                        ])


    CAT_COLS: List[str] = field(default_factory=lambda:['Current Military', 'Former Military','Never Military', 
                                                        'Asian', 'Black/African American', 'Other', 'Two or More', 'White',
                                                        'Born Citizen', 'Naturalized', 'Not a citizen',
                                                        ])
    
    CAT_DIMS: List[int] = field(default_factory=lambda:[3, 5, 3])  # Military has 3 categories, race has 5, citizenship has 3

    #target label for classification evaluation 
    LABEL: str = "Health Insurance"

@dataclass(frozen=True)
class HyperParams_HCUP(HyperParams):
  
    #list of numeric columns (IN ORDER)
    NUM_COLS: List[str] = field(default_factory=lambda: ['Age', 'Total Charge', 'Length of Stay', 'Number of Comorbidities'])


    #list of categorical columns (IN ORDER)
    BIN_COLS: List[str] = field(default_factory=lambda:['30 Day Readmission', 'Female', 'Urban', 'Teaching', 'Elective Admission',
                                                        'Acquired immune deficiency syndrome', 'Alcohol abuse',
                                                        'Anemias due to other nutritional deficiencies',
                                                        'Autoimmune conditions', 'Chronic blood loss', 'Leukemia', 'Lymphoma',
                                                        'Metastatic cancer', 'Solid tumor without metastasis in situ',
                                                        'Solid tumor without metastasis malignant', 'Cerebrovascular Disease on Admission',
                                                        'Cerebrovascular Disease Sequela', 'Coagulopathy', 'Dementia',
                                                        'Depression', 'Diabetes with chronic complications',
                                                        'Diabetes without chronic complications', 'Drug abuse', 'Heart failure',
                                                        'Hypertension complicated', 'Hypertension uncomplicated',
                                                        'Liver disease mild', 'Liver disease and failure moderate to severe',
                                                        'Chronic pulmonary disease', 'Neurological disorders affecting movement',
                                                        'Other neurological disorders', 'Seizures and epilepsy', 'Obesity',
                                                        'Paralysis', 'Peripheral vascular disease', 'Psychoses',
                                                        'Pulmonary circulation disease', 'Renal failure and disease moderate',
                                                        'Renal failure and disease severe', 'Hypothyroidism',
                                                        'Other thyroid disorders', 'Peptic ulcer with bleeding',
                                                        'Valvular disease', 'Weight loss'])
    CAT_COLS: List[str] = field(default_factory=lambda:['ZIP Income Quartile 1', 'ZIP Income Quartile 2', 'ZIP Income Quartile 3', 'ZIP Income Quartile 4', 'Payer Medicare', 'Payer Medicaid', 'Payer Private Insurance', 'Payer Self-pay', 'Payer No charge', 'Payer Other', 'Hospital Bedsize Small', 'Hospital Bedsize Medium', 'Hospital Bedsize Large'])
    CAT_DIMS: List[int] = field(default_factory=lambda:[4, 6, 3])  # 4 Income Quartiles, 6 insurance types, 3 hospital sizes
    LABEL: str = "30 Day Readmission"


@dataclass(frozen=True)
class HyperParams_MIMIC(HyperParams):
  
    #list of numeric columns (IN ORDER)
    NUM_COLS: List[str] = field(default_factory=lambda: [   'Age', 'alanine aminotransferase', 'albumin',
                                                            'alkaline phosphate', 'anion gap', 'asparate aminotransferase',
                                                            'basophils', 'bicarbonate', 'bilirubin', 'blood urea nitrogen', 'co2',
                                                            'co2 secondary', 'calcium', 'calcium ionized',
                                                            'central venous pressure', 'chloride', 'creatinine',
                                                            'diastolic blood pressure', 'fibrinogen', 'fraction inspired oxygen',
                                                            'fraction inspired oxygen set', 'glascow coma scale total', 'glucose',
                                                            'heart rate', 'height', 'hematocrit', 'hemoglobin', 'lactate',
                                                            'lactate dehydrogenase', 'lactic acid', 'lymphocytes', 'magnesium',
                                                            'mean blood pressure', 'mean corpuscular hemoglobin',
                                                            'mean corpuscular hemoglobin concentration', 'mean corpuscular volume',
                                                            'monocytes', 'neutrophils', 'oxygen saturation',
                                                            'partial pressure of carbon dioxide', 'partial pressure of oxygen',
                                                            'partial thromboplastin time', 'peak inspiratory pressure', 'phosphate',
                                                            'phosphorous', 'plateau pressure', 'platelets',
                                                            'positive end-expiratory pressure',
                                                            'positive end-expiratory pressure set', 'potassium', 'potassium serum',
                                                            'prothrombin time inr', 'prothrombin time pt', 'red blood cell count',
                                                            'respiratory rate', 'respiratory rate set', 'sodium',
                                                            'systolic blood pressure', 'temperature', 'tidal volume observed',
                                                            'tidal volume set', 'tidal volume spontaneous', 'troponin-t', 'weight',
                                                            'white blood cell count', 'ph', 'ph urine'])


    #list of categorical columns (IN ORDER)
    BIN_COLS: List[str] = field(default_factory=lambda: ['Male', 'mortality', 'vent', 'vaso', 'crystalloid_bolus', 'nivdurations'])


    CAT_COLS: List[str] = field(default_factory=lambda:['Black/African American', 'White', 'Other Race',
                                                        'Private Insurance', 'Uninsured', 'Public Insurance',
                                                        'Elective Admission', 'Emergency Admission', 'Urgent Admission'
                                                        ])
    CAT_DIMS: List[int] = field(default_factory=lambda:[3, 3, 3])  # Race has 3 categories, insurance has 3, admission type has 3

    #target label for classification evaluation 
    LABEL: str = "mortality"