"""
ENGRAM Mock Engine
Provides realistic MedGemma-style responses for local development
and demo recording without requiring a GPU.

For the demo video and local testing — swap to real MedGemma on Kaggle.
"""

from __future__ import annotations

import random

from PIL import Image

from .medgemma import BoundingBox, FeedbackResult, pad_image_to_square


# ─── Clinical Knowledge Base ──────────────────────────────────────
# Realistic findings and feedback per category

CLINICAL_DATA = {
    "Cardiomegaly": {
        "findings": [
            "Enlarged cardiac silhouette with cardiothoracic ratio > 0.5",
            "The heart shadow extends beyond the expected boundaries",
            "Left ventricular prominence suggesting cardiomegaly",
        ],
        "locations": [
            "Central mediastinum, left heart border displaced laterally",
            "Cardiac silhouette occupying >50% of thoracic diameter",
        ],
        "differentials": [
            "Dilated cardiomyopathy", "Pericardial effusion",
            "Valvular heart disease", "Hypertensive heart disease",
        ],
        "teaching": [
            "The cardiothoracic ratio (CTR) is measured on a PA chest X-ray. "
            "Draw a line across the widest point of the heart and divide by the "
            "widest internal thoracic diameter. A CTR > 0.5 indicates cardiomegaly. "
            "Always check if this is a PA vs AP film — AP films magnify the heart.",
            "Look for associated findings: pulmonary venous congestion (upper lobe "
            "diversion), pleural effusions, Kerley B lines. These suggest heart failure "
            "as the cause of cardiomegaly.",
        ],
        "boxes": [
            BoundingBox(y0=200, x0=300, y1=700, x1=700, label="cardiac_silhouette"),
            BoundingBox(y0=150, x0=250, y1=750, x1=750, label="enlarged_heart"),
        ],
    },
    "Pneumothorax": {
        "findings": [
            "Visible visceral pleural line with absent lung markings peripherally",
            "Lucency at the right apex with no vascular markings",
            "Thin white pleural line separated from the chest wall",
        ],
        "locations": [
            "Right apical region, between the visceral pleura and chest wall",
            "Left lateral hemithorax with collapsed lung medially",
        ],
        "differentials": [
            "Simple pneumothorax", "Tension pneumothorax",
            "Skin fold artifact", "Bullous emphysema",
        ],
        "teaching": [
            "The key finding is the visceral pleural line — a thin white line parallel "
            "to the chest wall with NO lung markings peripheral to it. Don't confuse "
            "this with skin folds, which typically extend beyond the lung apex and have "
            "lung markings on both sides.",
            "In tension pneumothorax, look for mediastinal shift AWAY from the affected "
            "side, tracheal deviation, and flattening of the ipsilateral hemidiaphragm. "
            "This is a clinical emergency requiring immediate needle decompression.",
        ],
        "boxes": [
            BoundingBox(y0=50, x0=600, y1=400, x1=950, label="pneumothorax"),
            BoundingBox(y0=80, x0=650, y1=350, x1=900, label="pleural_line"),
        ],
    },
    "Pleural Effusion": {
        "findings": [
            "Blunting of the costophrenic angle with meniscus sign",
            "Homogeneous opacity at the lung base obscuring the hemidiaphragm",
            "Layering fluid in the dependent portion of the thorax",
        ],
        "locations": [
            "Right costophrenic angle and lower hemithorax",
            "Bilateral basilar opacities with meniscus configuration",
        ],
        "differentials": [
            "Transudative effusion (heart failure)", "Exudative effusion (infection/malignancy)",
            "Hemothorax", "Empyema",
        ],
        "teaching": [
            "On an upright PA film, as little as 200mL of fluid can blunt the "
            "costophrenic angle. The meniscus sign — fluid climbing higher laterally "
            "than medially — is classic for free-flowing effusion. On a lateral film, "
            "the posterior costophrenic angle blunts first (75mL detectable).",
            "Large effusions can cause mediastinal shift TOWARD the opposite side. "
            "If the mediastinum shifts TOWARD the effusion, suspect associated "
            "atelectasis or an obstructing mass. Always compare with prior imaging.",
        ],
        "boxes": [
            BoundingBox(y0=600, x0=500, y1=950, x1=950, label="pleural_effusion"),
            BoundingBox(y0=650, x0=50, y1=950, x1=450, label="costophrenic_blunting"),
        ],
    },
    "Lung Opacity": {
        "findings": [
            "Patchy airspace opacity in the right middle lobe",
            "Ill-defined area of increased density in the lung parenchyma",
            "Confluent opacity with air bronchograms",
        ],
        "locations": [
            "Right middle lobe, obscuring the right heart border (silhouette sign)",
            "Left lower lobe posterior segment",
        ],
        "differentials": [
            "Pneumonia", "Pulmonary edema", "Hemorrhage",
            "Organizing pneumonia", "Lung mass",
        ],
        "teaching": [
            "The silhouette sign helps localize opacities: if the right heart border "
            "is obscured, the opacity is in the right middle lobe. If the right "
            "hemidiaphragm is obscured, it's in the right lower lobe. Use this "
            "systematically to determine the anatomical location.",
            "Air bronchograms — air-filled bronchi visible within an opacity — "
            "indicate the surrounding alveoli are filled with fluid/pus/cells. "
            "This is a hallmark of consolidation and helps distinguish it from "
            "atelectasis (which has volume loss).",
        ],
        "boxes": [
            BoundingBox(y0=250, x0=100, y1=550, x1=400, label="lung_opacity"),
        ],
    },
    "Consolidation": {
        "findings": [
            "Dense homogeneous opacity with air bronchograms",
            "Lobar consolidation with sharp fissural margin",
            "Complete opacification of the right lower lobe",
        ],
        "locations": [
            "Right lower lobe with clear border along the major fissure",
            "Left upper lobe lingular segment",
        ],
        "differentials": [
            "Bacterial pneumonia", "Aspiration pneumonia",
            "Pulmonary hemorrhage", "Organizing pneumonia",
        ],
        "teaching": [
            "Consolidation means the airspaces are filled with material (pus, fluid, "
            "blood, cells). The hallmark is air bronchograms — the bronchi remain "
            "air-filled while surrounding alveoli are opacified. Unlike atelectasis, "
            "there is typically no volume loss.",
        ],
        "boxes": [
            BoundingBox(y0=350, x0=550, y1=700, x1=900, label="consolidation"),
        ],
    },
    "Atelectasis": {
        "findings": [
            "Linear band-like opacity with volume loss",
            "Elevation of the hemidiaphragm and mediastinal shift toward the opacity",
            "Crowding of the ribs on the affected side",
        ],
        "locations": [
            "Left lower lobe with elevated left hemidiaphragm",
            "Right upper lobe with upward displacement of the minor fissure",
        ],
        "differentials": [
            "Obstructive atelectasis (mass, mucus plug)",
            "Compressive atelectasis (pleural effusion)",
            "Adhesive atelectasis (surfactant deficiency)",
        ],
        "teaching": [
            "The key distinguishing feature of atelectasis vs consolidation is "
            "VOLUME LOSS. Look for: elevated hemidiaphragm, mediastinal shift "
            "toward the opacity, fissure displacement, and rib crowding. If you "
            "see an opacity WITH volume loss = atelectasis. Without volume loss = "
            "consolidation.",
        ],
        "boxes": [
            BoundingBox(y0=400, x0=100, y1=700, x1=450, label="atelectasis"),
        ],
    },
    "Edema": {
        "findings": [
            "Bilateral perihilar haziness with upper lobe pulmonary venous distension",
            "Kerley B lines at the lung periphery",
            "Bilateral pleural effusions with peribronchial cuffing",
        ],
        "locations": [
            "Bilateral, symmetric, predominantly perihilar (bat-wing pattern)",
            "Diffuse bilateral airspace opacity in gravity-dependent distribution",
        ],
        "differentials": [
            "Cardiogenic pulmonary edema", "ARDS",
            "Fluid overload", "Renal failure",
        ],
        "teaching": [
            "The classic progression of pulmonary edema on CXR: first, upper lobe "
            "venous distension (cephalization). Then, interstitial edema (Kerley B "
            "lines, peribronchial cuffing). Finally, alveolar edema (bat-wing "
            "pattern). Pleural effusions are common. Always check the heart size — "
            "cardiogenic edema usually has cardiomegaly; ARDS does not.",
        ],
        "boxes": [
            BoundingBox(y0=200, x0=200, y1=600, x1=800, label="pulmonary_edema"),
            BoundingBox(y0=100, x0=350, y1=300, x1=650, label="cephalization"),
        ],
    },
    "Fracture": {
        "findings": [
            "Cortical disruption of the lateral right rib",
            "Step-off deformity visible along the rib contour",
            "Displaced fracture fragment with adjacent soft tissue swelling",
        ],
        "locations": [
            "Right lateral 6th and 7th ribs",
            "Left posterior 9th rib",
        ],
        "differentials": [
            "Acute traumatic fracture", "Pathologic fracture (metastasis)",
            "Stress fracture", "Old healed fracture",
        ],
        "teaching": [
            "Rib fractures are easily missed on frontal CXR. Trace each rib "
            "systematically from posterior to anterior. Look for cortical "
            "disruptions, step-offs, and displacement. The lower ribs (8-12) "
            "are associated with splenic/hepatic injury. Always count ribs "
            "and document locations precisely.",
        ],
        "boxes": [
            BoundingBox(y0=350, x0=700, y1=450, x1=900, label="rib_fracture"),
        ],
    },
    "No Finding": {
        "findings": [
            "No acute cardiopulmonary abnormality",
            "Clear lung fields bilaterally",
            "Normal heart size and mediastinal contour",
        ],
        "locations": [
            "N/A — no focal abnormality identified",
        ],
        "differentials": [
            "Normal study — no differential needed",
        ],
        "teaching": [
            "Even on a 'normal' film, practice your systematic approach: check "
            "airways (trachea midline?), bones (fractures?), cardiac (size, contour), "
            "diaphragm (flattening, free air?), everything else (soft tissues, tubes). "
            "The most dangerous reading in radiology is 'normal' — it means you "
            "looked at everything and found nothing. Make sure you actually looked.",
        ],
        "boxes": [],
    },
    "Pneumonia": {
        "findings": [
            "Focal consolidation with air bronchograms in the right lower lobe",
            "Patchy bilateral infiltrates with ground-glass opacity",
        ],
        "locations": [
            "Right lower lobe posterior basal segment",
            "Bilateral multifocal involvement",
        ],
        "differentials": [
            "Community-acquired pneumonia", "Aspiration pneumonia",
            "Viral pneumonia", "Opportunistic infection",
        ],
        "teaching": [
            "Bacterial pneumonia typically presents as lobar consolidation — dense, "
            "homogeneous opacity confined to one lobe. Viral/atypical pneumonia "
            "shows more diffuse, patchy, bilateral interstitial or ground-glass "
            "pattern. Clinical context matters: acute onset + fever + lobar = "
            "likely bacterial. Gradual onset + bilateral = consider viral/atypical.",
        ],
        "boxes": [
            BoundingBox(y0=400, x0=550, y1=750, x1=900, label="pneumonia"),
        ],
    },
    "Support Devices": {
        "findings": [
            "Endotracheal tube with tip 3cm above the carina",
            "Central venous catheter via right internal jugular with tip in the SVC",
            "Nasogastric tube with tip in the stomach",
        ],
        "locations": [
            "ETT midline, tip projecting over T3-T4",
            "CVC tip at the cavoatrial junction",
        ],
        "differentials": [
            "Correctly positioned lines/tubes",
            "Malpositioned ETT (too high/low)",
            "Malpositioned central line (in IJ, subclavian, or azygos)",
        ],
        "teaching": [
            "Always check line/tube positions on every ICU film. ETT tip should be "
            "3-5cm above the carina (approximately T3-T4). Central line tip should "
            "be at the cavoatrial junction. NG tube should follow the midline to "
            "below the left hemidiaphragm. A malpositioned tube can kill — this is "
            "not optional reading.",
        ],
        "boxes": [
            BoundingBox(y0=50, x0=450, y1=400, x1=550, label="ett_tube"),
        ],
    },
}

# Default for categories not in the knowledge base
DEFAULT_CLINICAL = {
    "findings": ["Abnormal finding identified in this region"],
    "locations": ["Central thoracic region"],
    "differentials": ["Further correlation needed"],
    "teaching": [
        "Systematic review of the chest X-ray is essential. Use the ABCDE "
        "approach: Airways, Bones, Cardiac, Diaphragm, Everything else."
    ],
    "boxes": [BoundingBox(y0=300, x0=300, y1=700, x1=700, label="finding")],
}


class MockMedGemmaEngine:
    """
    Mock engine that returns clinically accurate responses
    without requiring a GPU. Uses the knowledge base above.
    """

    def __init__(self):
        self._loaded = True

    def load(self):
        pass

    def analyze_image(self, image: Image.Image, category: str = "") -> str:
        """Return a realistic clinical interpretation."""
        data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
        finding = random.choice(data["findings"])
        location = random.choice(data["locations"])
        differentials = ", ".join(random.sample(data["differentials"], min(3, len(data["differentials"]))))

        return (
            f"**Findings:** {finding}\n\n"
            f"**Location:** {location}\n\n"
            f"**Differential Diagnosis:** {differentials}\n\n"
            f"**Recommendation:** Clinical correlation recommended. "
            f"Comparison with prior imaging if available."
        )

    def localize_findings(self, image: Image.Image, category: str = "") -> list[BoundingBox]:
        """Return realistic bounding boxes for the category."""
        data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
        return data["boxes"]

    def generate_question(
        self,
        image: Image.Image,
        category: str,
        difficulty: str = "intermediate",
    ) -> str:
        """Generate a realistic clinical question."""
        vignettes = {
            "beginner": [
                ("65-year-old male", "progressive dyspnea and orthopnea for 2 weeks"),
                ("45-year-old female", "acute chest pain and shortness of breath"),
                ("72-year-old male", "productive cough and fever for 3 days"),
                ("58-year-old female", "routine pre-operative chest X-ray"),
            ],
            "intermediate": [
                ("55-year-old male with COPD", "worsening dyspnea and right-sided chest pain"),
                ("38-year-old female post-thyroidectomy", "new onset stridor and hypoxia"),
                ("67-year-old diabetic male", "fever, confusion, and productive cough"),
                ("44-year-old female on warfarin", "sudden pleuritic chest pain"),
            ],
            "advanced": [
                ("23-year-old male post-MVA", "chest pain, tachycardia, hypotension, JVD"),
                ("61-year-old female with lupus", "progressive dyspnea, muffled heart sounds"),
                ("49-year-old male post-central line placement", "sudden desaturation"),
                ("34-year-old female, immunosuppressed", "dry cough, bilateral infiltrates"),
            ],
        }

        patient, presentation = random.choice(vignettes.get(difficulty, vignettes["intermediate"]))

        return (
            f"**Clinical Vignette:**\n"
            f"A {patient} presents with {presentation}.\n\n"
            f"**Task:**\n"
            f"1. Identify the key finding(s) on this chest X-ray\n"
            f"2. Describe the anatomical location precisely\n"
            f"3. Provide your differential diagnosis\n"
            f"4. What additional imaging or tests would you recommend?"
        )

    # ─── F2: Socratic Mode ──────────────────────────────────────
    SOCRATIC_TEMPLATES = {
        "Cardiomegaly": {
            "cardiothoracic": "What quantitative measurement on a PA film helps assess heart size?",
            "ctr": "Can you tell me the normal range for the cardiothoracic ratio?",
            "enlarged": "What borders of the heart would you examine first?",
        },
        "Pneumothorax": {
            "pleural line": "What key finding separates pneumothorax from a skin fold?",
            "absent lung markings": "What should you see peripheral to the visceral pleural line?",
            "lucency": "Where should you look first for a small pneumothorax?",
        },
        "Pleural Effusion": {
            "meniscus": "What shape does free-flowing fluid form against the chest wall?",
            "blunting": "Which angle on a chest X-ray is first blunted by fluid?",
            "costophrenic": "How much fluid is needed to blunt the costophrenic angle on a PA film?",
        },
        "Lung Opacity": {
            "silhouette": "If the right heart border is obscured, which lobe is the opacity in?",
            "air bronchogram": "What do air-filled bronchi within an opacity tell you?",
            "opacity": "How would you distinguish consolidation from atelectasis?",
        },
        "Consolidation": {
            "air bronchogram": "What finding within the opacity confirms alveolar filling?",
            "lobar": "What structure creates a sharp border for lobar consolidation?",
            "volume loss": "Is there volume loss here? What does its absence tell you?",
        },
        "Atelectasis": {
            "volume loss": "What signs indicate volume loss on a chest X-ray?",
            "collapse": "Which direction does the mediastinum shift in atelectasis?",
            "fissure": "How do the fissures move in upper lobe versus lower lobe collapse?",
        },
        "Edema": {
            "cephalization": "What happens to the upper lobe vessels in early pulmonary edema?",
            "kerley": "What are Kerley B lines and what do they represent pathologically?",
            "bat wing": "What pattern does alveolar edema typically create on CXR?",
        },
        "Fracture": {
            "cortical": "What specific finding along the rib contour confirms a fracture?",
            "displacement": "Why is it important to count and document rib fracture locations?",
            "fracture": "Which lower rib fractures are associated with abdominal organ injury?",
        },
        "Pneumonia": {
            "consolidation": "How does bacterial pneumonia differ from viral on imaging?",
            "infiltrate": "What clinical features help distinguish pneumonia from pulmonary edema?",
            "air bronchogram": "What does the presence of air bronchograms tell you?",
        },
        "No Finding": {
            "normal": "Walk me through your systematic approach for reading this film.",
            "clear": "What specific structures did you check before calling this normal?",
            "unremarkable": "What is the most dangerous thing about calling a film 'normal'?",
        },
        "Support Devices": {
            "tube": "Where should the tip of an endotracheal tube be positioned?",
            "line": "What is the ideal location for a central venous catheter tip?",
            "catheter": "What are the consequences of a malpositioned NG tube?",
        },
    }

    def generate_socratic_question(
        self, image: Image.Image, student_answer: str, category: str,
    ) -> str:
        """Generate Socratic probing question based on what student missed."""
        answer_lower = (student_answer or "").lower()
        templates = self.SOCRATIC_TEMPLATES.get(category, {})

        # Find keywords the student missed
        for keyword, question in templates.items():
            if keyword not in answer_lower:
                return f"**Socratic Question:**\n\n{question}\n\n*Think about this before seeing the full answer.*"

        return "**Good coverage!** Can you think of any additional findings or differential diagnoses?"

    def generate_socratic_followup(
        self, student_answer: str, socratic_response: str, category: str,
    ) -> str:
        """Evaluate student's Socratic response."""
        data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
        response_lower = (socratic_response or "").lower()

        keywords = []
        for finding in data["findings"]:
            keywords.extend(w for w in finding.lower().split() if len(w) > 4)

        matches = sum(1 for kw in keywords[:6] if kw in response_lower)
        if matches >= 2:
            return "**Correct!** You're on the right track. Let's see the full analysis."
        elif matches >= 1:
            return "**Getting there.** You've identified part of it. Here's the complete picture."
        else:
            hint = data["teaching"][0][:120] + "..."
            return f"**Hint:** {hint}\n\nLet me show you the full expert analysis."

    # ─── F3: Satisfaction of Search ──────────────────────────────

    def grade_search_completeness(
        self, student_answer: str, category: str,
    ) -> tuple[list[str], list[str], float]:
        """Grade whether student found ALL findings.
        Returns (found_findings, missed_findings, completeness_score).
        """
        data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
        all_findings = data["findings"]
        answer_lower = (student_answer or "").lower()

        found = []
        missed = []
        for finding in all_findings:
            # Extract key terms (words > 4 chars)
            key_terms = [w.lower() for w in finding.split() if len(w) > 4]
            if any(term in answer_lower for term in key_terms[:3]):
                found.append(finding[:60])
            else:
                missed.append(finding[:60])

        total = len(all_findings)
        completeness = len(found) / total if total > 0 else 0.0
        return found, missed, completeness

    # ─── F4: Dual-Process Gestalt Grading ──────────────────────

    GESTALT_KEYWORDS = {
        "Cardiomegaly": ["heart", "enlarged", "big", "cardio", "large"],
        "Pneumothorax": ["pneumo", "air", "collapsed", "lung"],
        "Pleural Effusion": ["fluid", "effusion", "white", "base"],
        "Lung Opacity": ["opacity", "white", "hazy", "shadow"],
        "Consolidation": ["consolidation", "solid", "dense", "white"],
        "Atelectasis": ["collapse", "atelectasis", "volume"],
        "Edema": ["edema", "fluid", "hazy", "bilateral"],
        "Fracture": ["fracture", "broken", "rib", "break"],
        "No Finding": ["normal", "clear", "nothing", "unremarkable"],
        "Pneumonia": ["pneumonia", "infection", "consolidation"],
        "Support Devices": ["tube", "line", "device", "wire"],
    }

    def grade_gestalt(self, student_answer: str, category: str) -> float:
        """Grade a rapid gestalt impression (System 1). Lenient keyword matching."""
        answer_lower = (student_answer or "").lower()
        if not answer_lower.strip():
            return 0.0
        keywords = self.GESTALT_KEYWORDS.get(category, [category.lower()])
        matches = sum(1 for kw in keywords if kw in answer_lower)
        if matches >= 2:
            return 0.85 + random.uniform(0, 0.15)
        elif matches >= 1:
            return 0.55 + random.uniform(0, 0.2)
        return 0.1 + random.uniform(0, 0.15)

    # ─── F5: Contrastive Case Pairs ─────────────────────────────

    CONTRASTIVE_PAIRS = {
        ("Consolidation", "Atelectasis"): {
            "question": "Both images show opacification. One is consolidation, the other atelectasis. What is the KEY distinguishing feature?",
            "key_difference": "Volume loss. Atelectasis has volume loss (elevated diaphragm, mediastinal shift toward opacity, fissure displacement). Consolidation does not.",
            "keywords": ["volume loss", "shift", "collapse", "fissure", "diaphragm"],
        },
        ("Pleural Effusion", "Lung Opacity"): {
            "question": "Both show whiteness at the base. One is effusion, the other parenchymal opacity. How do you tell them apart?",
            "key_difference": "Meniscus sign. Effusions are gravity-dependent with a curved upper border (meniscus). Parenchymal opacities don't layer with position change.",
            "keywords": ["meniscus", "gravity", "layer", "decubitus", "costophrenic"],
        },
        ("Cardiomegaly", "Edema"): {
            "question": "Both cases involve the heart and lungs. One shows cardiomegaly alone, the other pulmonary edema. What features distinguish them?",
            "key_difference": "Cardiomegaly is heart-size only (CTR > 0.5). Edema shows lung findings: cephalization, Kerley B lines, bilateral haziness, peribronchial cuffing.",
            "keywords": ["cephalization", "kerley", "lung", "ctr", "haziness"],
        },
        ("Pneumothorax", "No Finding"): {
            "question": "One of these films has a subtle pneumothorax. The other is normal. Can you identify which is which and why?",
            "key_difference": "Look for the visceral pleural line — a thin white line parallel to the chest wall with absent lung markings beyond it. Normal films have vascular markings extending to the periphery.",
            "keywords": ["pleural line", "lung markings", "peripheral", "visceral"],
        },
        ("Pneumonia", "Consolidation"): {
            "question": "Both show dense opacification. One is typical bacterial pneumonia, the other a non-infectious consolidation. What clinical and imaging clues help?",
            "key_difference": "Imaging alone often cannot distinguish them. Clinical context is key: fever + acute onset suggests infection. Air bronchograms appear in both. Follow-up imaging showing resolution with antibiotics confirms pneumonia.",
            "keywords": ["clinical", "fever", "follow-up", "resolution", "antibiotics"],
        },
        ("Edema", "Pneumonia"): {
            "question": "Both show bilateral opacities. One is pulmonary edema, the other bilateral pneumonia. What patterns help distinguish them?",
            "key_difference": "Edema is typically symmetric, perihilar (bat-wing), with cephalization and often cardiomegaly. Pneumonia tends to be asymmetric, lobar, with air bronchograms and no cephalization.",
            "keywords": ["symmetric", "perihilar", "asymmetric", "cardiomegaly", "cephalization"],
        },
    }

    def generate_contrastive_question(self, category_a: str, category_b: str) -> str:
        """Generate a contrastive comparison question."""
        pair = self.CONTRASTIVE_PAIRS.get(
            (category_a, category_b),
            self.CONTRASTIVE_PAIRS.get((category_b, category_a)),
        )
        if pair:
            return pair["question"]
        return f"Compare these two cases. One is {category_a}, the other is {category_b}. What is the key distinguishing feature?"

    def grade_contrastive(
        self, student_answer: str, category_a: str, category_b: str,
    ) -> FeedbackResult:
        """Grade contrastive pair discrimination."""
        pair = self.CONTRASTIVE_PAIRS.get(
            (category_a, category_b),
            self.CONTRASTIVE_PAIRS.get((category_b, category_a)),
        )
        answer_lower = (student_answer or "").lower()

        if pair:
            keywords = pair["keywords"]
            matches = sum(1 for kw in keywords if kw in answer_lower)
            if matches >= 2:
                score = 0.8 + random.uniform(0, 0.2)
            elif matches >= 1:
                score = 0.5 + random.uniform(0, 0.2)
            else:
                score = 0.1 + random.uniform(0, 0.2)
            explanation = f"**Key Difference:** {pair['key_difference']}"
            correct = [kw for kw in keywords if kw in answer_lower]
            missed = [kw for kw in keywords if kw not in answer_lower]
        else:
            score = 0.5
            explanation = f"Compare the characteristic features of {category_a} versus {category_b}."
            correct, missed = [], []

        return FeedbackResult(
            score=min(1.0, score),
            correct_findings=correct,
            missed_findings=missed[:3],
            false_positives=[],
            explanation=explanation,
            box_iou=0.0,
        )

    # ─── Core grading ────────────────────────────────────────────

    def grade_response(
        self,
        image: Image.Image,
        student_answer: str,
        category: str,
        ground_truth: str = "",
    ) -> FeedbackResult:
        """Grade a student's response with realistic feedback."""
        data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
        answer_lower = student_answer.lower() if student_answer else ""

        # Simple keyword matching for scoring
        category_keywords = {
            "Cardiomegaly": ["cardiomegaly", "enlarged heart", "ctr", "cardiothoracic"],
            "Pneumothorax": ["pneumothorax", "pleural line", "absent lung markings", "lucency"],
            "Pleural Effusion": ["effusion", "blunting", "meniscus", "costophrenic"],
            "Lung Opacity": ["opacity", "infiltrate", "airspace", "silhouette"],
            "Consolidation": ["consolidation", "air bronchogram", "lobar"],
            "Atelectasis": ["atelectasis", "collapse", "volume loss"],
            "Edema": ["edema", "cephalization", "kerley", "bat wing"],
            "Pneumonia": ["pneumonia", "infiltrate", "consolidation"],
            "Fracture": ["fracture", "cortical", "break", "displacement"],
            "No Finding": ["normal", "no finding", "unremarkable", "clear"],
            "Support Devices": ["tube", "line", "catheter", "ett", "cvp"],
        }

        keywords = category_keywords.get(category, [category.lower()])
        matches = sum(1 for kw in keywords if kw in answer_lower)

        if not answer_lower.strip():
            score = 0.0
        elif matches >= 2:
            score = 0.85 + random.uniform(0, 0.15)
        elif matches >= 1:
            score = 0.55 + random.uniform(0, 0.2)
        else:
            score = 0.15 + random.uniform(0, 0.2)

        finding = random.choice(data["findings"])
        teaching = random.choice(data["teaching"])

        correct = [kw for kw in keywords if kw in answer_lower]
        missed = [kw for kw in keywords if kw not in answer_lower]

        return FeedbackResult(
            score=min(1.0, score),
            correct_findings=correct if correct else [],
            missed_findings=missed[:3],
            false_positives=[],
            explanation=(
                f"**Ground Truth:** {finding}\n\n"
                f"**Teaching Point:**\n{teaching}"
            ),
            box_iou=score * 0.8,  # Approximate spatial accuracy from score
        )


