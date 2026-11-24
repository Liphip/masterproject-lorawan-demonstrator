# IoT & AI Demonstrator — Assignment Requirements

**University of Kassel · ComTec · Master's Project**  
**Date:** 23 November 2026  
**Reference:** *Studentische Arbeit: Erstellen eines IoT KI Demonstrators* (25 March 2026)  
**Repository:** University GitLab (mirror: [github.com/Liphip/masterproject-lorawan-demonstrator](https://github.com/Liphip/masterproject-lorawan-demonstrator))  
**Status:** Assignment scope and intended outcome; not a status report. Current status: [PLAN.md](PLAN.md).

## 1. Objective

Develop a **portable demonstrator** that makes the interaction between the Internet of Things (IoT), LoRaWAN and artificial intelligence (AI) understandable through a tangible example. Real LoRaWAN sensors transmit measurements through a gateway and a router to a laptop. The laptop processes the data, runs machine-learning (ML) models and visualizes the results.

The demonstrator is intended for presentations at the university and to external audiences. The emphasis is on a **working, understandable end-to-end demonstration**, rather than on building a general-purpose IoT platform or production software.

The original assignment refers to the *“Ahle Wurst meets KI”* use case. The project additionally aims to demonstrate **several different, computationally modest ML approaches side by side**, showing how the same sensor data can lead to different model outputs and trade-offs.

## 2. Intended demonstrator setup

**System path:** LoRaWAN sensors → **Milesight UG63 gateway** → **router / Wi-Fi network** → **laptop** (data reception → preprocessing → CPU-based ML inference → comparative visualization).

The gateway–router–laptop connection **symbolically represents the Internet/network segment**, as requested by the assignment. MQTT is a plausible transport mechanism, but the original brief does **not** mandate MQTT or any particular dashboard technology.

The physical hardware is already available. The planned setup uses the existing laptop, **Milesight UG63 gateway**, router/Wi-Fi equipment, LoRaWAN sensors and transport case. The case holds a **Milesight WS202** (PIR/light), a **Dragino LSE01** (moisture, conductivity, temperature) and a **Dragino LTC-FS** (temperature probe): the same sensor types as in the training data. (The sketch in the March brief shows an LSN50-v2-S31 instead; it is not part of the case.) Power-supply models should be documented during final assembly.

## 3. Requirements for the assignment

The following are **outcome-oriented requirements**. *Core* items follow the original assignment; *Project extension* items reflect the additional ML comparison goal. *Recommended* items help make the demonstration reliable and scientifically meaningful without turning it into a large software project.

| ID | Priority | Requirement |
|---|---|---|
| **A1** | Core | Build a small, functioning LoRaWAN system using the existing sensor devices and gateway. Measurements must reach the laptop through the router/network arrangement. |
| **A2** | Core | Show sensor measurements in a human-readable form, including the sensor/source and relevant measured values. |
| **A3** | Core | Process sensor data with AI/ML **on the laptop**, rather than delegating the demonstrated inference to an external cloud service. |
| **A4** | Core | Provide a visualization that makes both the incoming IoT data and the AI result intelligible to an audience. |
| **A5** | Core | Assemble the components into a transportable case; enable practical setup, operation and repacking for demonstrations at different locations. |
| **A6** | Project extension | Demonstrate **at least three distinct, lightweight ML model families** side by side for a shared, clearly defined prediction task. |
| **A7** | Project extension | Feed the models comparable data and display their predictions together. Make similarities, disagreements and relevant trade-offs understandable. |
| **A8** | Project extension | Ensure that the deployed models are feasible for **CPU-only inference** on the demonstration laptop, with reasonable compute and power demand. A GPU must not be required during presentations. |
| **A9** | Recommended | Support a recorded-data or replay demonstration in addition to live sensor reception, particularly for processes that evolve over hours or days. The display must distinguish live and replayed data. |
| **A10** | Recommended | Make startup, basic operation and shutdown repeatable for someone other than the developer using short instructions. |
| **A11** | Recommended | Explain the main limitations of the dataset, prediction task and model quality; do not present model outputs as ground truth. |

### 3.1 Machine-learning comparison

The current repository provides a useful initial comparison across three model families:

| Approach | Purpose in the demonstration |
|---|---|
| **Logistic Regression** | Simple, interpretable baseline for classification. |
| **Random Forest** | Nonlinear tree-based model with moderate inference cost. |
| **Multilayer Perceptron (MLP)** | Compact neural-network approach, demonstrating a different modeling principle without requiring a GPU. |

The selected models should be evaluated on the **same prediction task**, with comparable input data, a consistent evaluation procedure and suitable measures such as accuracy, balanced accuracy or macro-F1. Where possible, separation by independent sausage/run should avoid evaluating a model on windows derived from the same recording used for its training.

A model's prediction, any meaningful confidence/score, and a basic comparison of quality or inference effort are more valuable for this assignment than an exhaustive hyperparameter search. Additional methods (such as a small boosting model) are optional, not a requirement. Different tasks—such as classification versus anomaly detection—must not be ranked using one common accuracy number.

**Important demonstration limitation:** Classification of sausage-curing phases is based on historical time-series behavior. Changing a live sensor reading for a few seconds cannot meaningfully simulate days of ripening. Therefore, the most credible presentation combines a **live LoRaWAN data path** with **recorded/replayed time-series data for meaningful model comparisons**. Live model predictions should only be displayed where the available data is actually compatible with the trained task.

### 3.2 Data processing and visualization

The application should take the relevant measurements from the gateway, prepare them in the representation required by the models and show their results. The repository already contains feature extraction, time-window processing and a custom scikit-learn pipeline abstraction. These may be reused rather than replaced with a new framework.

The visualization should answer four questions at a glance: **Which sensor sent data? What did it measure? What does each model predict? How do the predictions compare?** A simple dashboard, notebook-based interface or Grafana-based display is acceptable if it is clear and dependable. Sophisticated user management, analytics infrastructure and cloud hosting are not necessary.

## 4. Expected results and handover

The work is complete at the assignment level when the following results can be presented:

1. **Physical demonstrator:** existing hardware assembled, connected and safely transportable in the case.
2. **End-to-end demonstration:** a real LoRaWAN measurement reaches the laptop and is displayed; the gateway and router roles can be explained.
3. **AI demonstration:** three lightweight models can be run and their results compared side by side on a suitable common input.
4. **Credible example:** the connection to *“Ahle Wurst meets KI”* or another clearly defined sensor-data task is explained, including what the models actually predict.
5. **Basic evaluation:** a concise comparison of model behavior, prediction quality and CPU suitability, including relevant data limitations.
6. **Reproducible operation:** brief setup/usage instructions and the software/data needed for a future demonstration are available in the repository.

### Suggested final demonstration

**Step 1 — Hardware and data path:** Open the case, connect power, start the network and show at least one newly received sensor measurement on the laptop.

**Step 2 — AI and comparison:** Select a representative recorded sequence or suitable current observation and show the three model predictions simultaneously. Explain what their outputs mean and where they differ.

**Step 3 — Interpretation:** Show or summarize how the approaches compare on evaluation data and why a small CPU-capable model can be appropriate for a mobile IoT application.

**Step 4 — Portability:** Shut down and pack the demonstrator using the documented procedure.

## 5. Scope boundaries and current baseline

This is a **student demonstration assignment**, not a commission to produce an industrial IoT product. It does **not** require a microservice architecture, production-grade authentication, a comprehensive automated test suite, 24/7 operation, cloud deployment, mobile applications, or a broad survey of deep-learning methods. Training can be performed separately; the presentation-time inference must be practical on the laptop CPU. Internet independence is desirable if the installed gateway/network setup supports it, but it is not an explicit requirement of the original assignment.

**Baseline at the start of this scope (23 November 2026):** the repository contained historic training data, conversion/training notebooks, feature extraction, windowing and ML pipeline code. Logistic Regression, Random Forest and MLP were defined in the training notebook. A gateway ingestion application, an integrated comparison display and a deployment-ready demonstrator startup did not exist yet. The **hardware exists**, as confirmed for this project; its final assembly and operation have not been independently verified here.

The existing data files and notebook code are a starting point; completing the assignment means integrating and **demonstrating the whole chain**, not writing a much larger requirements or software architecture package.

---

*Source distinction: the March 2026 ComTec brief establishes the core assignment; the side-by-side algorithm comparison and CPU-only emphasis are later project goals. The baseline refers to the repository state on the date above. This document is a proposed concise assignment scope, not a supervisor-approved change to the original brief.*
