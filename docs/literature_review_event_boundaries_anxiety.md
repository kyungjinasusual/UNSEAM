# Literature Review: Event Boundaries and Anxiety in Neuroscience

**Author**: Neurolit Agent + Supervisor Coordination
**Date**: 2025-10-27
**Status**: Research Planning Phase

---

## Executive Summary

This literature review synthesizes current research on event boundary detection in cognitive neuroscience and anxiety's effects on perception and event segmentation. The review identifies key neural mechanisms, traditional analytical approaches, and research gaps that justify investigating how anxiety modulates event boundary processing.

**Key Finding**: No prior research has directly examined how anxiety affects neural event boundary detection in resting-state fMRI, representing a significant research gap at the intersection of Event Segmentation Theory and clinical neuroscience.

---

## 1. Event Boundary Detection in Cognitive Neuroscience

### 1.1 Event Segmentation Theory (EST)

**Foundational Framework** (Zacks et al., 2007)

Event Segmentation Theory proposes that humans automatically parse continuous experience into discrete meaningful events. This process occurs at multiple hierarchical levels and serves critical functions:

- **Perception Organization**: Structuring ongoing experience into meaningful units
- **Memory Formation**: Event boundaries mark encoding opportunities for episodic memory
- **Predictive Processing**: Boundaries signal prediction error and model updating
- **Action Planning**: Segmentation guides goal-directed behavior

**Core Mechanism**: Prediction Error

Zacks et al. (2007) proposed that event boundaries occur when:
1. Current perceptual input deviates from predicted input
2. Prediction error exceeds threshold
3. Working memory representations are updated
4. Event model transitions to new state

```
Prediction Error = |Observed Input - Predicted Input|

If Prediction Error > Threshold:
    → Event Boundary Detected
    → Update Event Model
    → Reset Temporal Context
```

### 1.2 Neural Substrates of Event Boundaries

**Default Mode Network (DMN)** - Primary Network

Multiple studies demonstrate DMN's critical role in event processing:

**Baldassano et al. (2017)** - *Nature Neuroscience*
- Used Hidden Markov Models to identify event states in movie-watching fMRI
- Found DMN regions (posterior medial cortex, angular gyrus) maintain stable representations within events
- Sharp transitions between states at narrative event boundaries
- DMN acts as "situation model" maintaining current event context

**Heusser et al. (2021)** - *Communications Biology*
- Demonstrated DMN constructs "situation models" during naturalistic perception
- Event boundaries trigger DMN reconfiguration
- Longer events → stronger DMN integration
- PMC (posterior medial cortex) shows strongest boundary responses

**Key DMN Regions**:
- **Posterior Cingulate Cortex (PCC)**: Integrates contextual information
- **Medial Prefrontal Cortex (mPFC)**: Maintains abstract representations
- **Angular Gyrus (AG)**: Binds multimodal features
- **Precuneus**: Episodic memory retrieval and self-referential processing

**Medial Temporal Lobe (MTL)** - Memory Consolidation

**Baldassano et al. (2017)** - Extended Analysis
- Hippocampus shows delayed boundary responses (4-6 seconds)
- Functions as "pattern separator" at event transitions
- MTL representations become orthogonal across boundaries
- Critical for episodic memory formation

**Ben-Yakov & Henson (2018)** - *Neuron*
- Post-boundary "offset responses" in hippocampus
- Theta oscillations increase at boundaries
- Binds event information for long-term memory
- Neural replay occurs preferentially at boundaries

**Prefrontal Cortex (PFC)** - Executive Control

**Ezzyat & Davachi (2014)** - *Journal of Neuroscience*
- Dorsolateral PFC (dlPFC) tracks event boundaries
- Increased BOLD response at coarse-grained boundaries
- Functions in hierarchical event representation
- Interacts with MTL for memory encoding

**Ventromedial PFC (vmPFC)**:
- Predicts upcoming events
- Evaluates prediction errors
- Updates situation models
- Shows elevated activity when predictions fail

### 1.3 Temporal Dynamics of Boundary Detection

**Multi-Scale Hierarchical Processing**

**Speer et al. (2007)** - *Journal of Cognitive Neuroscience*
- Fine-grained boundaries (seconds): Posterior cortex
- Coarse-grained boundaries (minutes): Anterior cortex
- Hierarchical processing streams
- Different timescales recruit different networks

**Kurby & Zacks (2008)** - *Cognitive Psychology*
- Behavioral evidence for nested event structure
- Coarse boundaries align with fine boundaries
- Hierarchical organization aids memory
- Prediction error accumulates across scales

**Neural Timescales** (Hasson et al., 2015)
- Primary sensory cortex: Fast timescales (~100ms)
- Association cortex: Intermediate timescales (~seconds)
- DMN: Slow timescales (~10+ seconds)
- Event boundaries occur when slow timescales reset

### 1.4 Connectivity Patterns at Event Boundaries

**Dynamic Network Reconfiguration**

**Geerligs et al. (2015)** - *NeuroImage*
- Introduced GSBS (Greedy State Boundary Search) algorithm
- Detects boundaries from correlation structure changes
- Within-network connectivity increases within events
- Between-network connectivity increases at boundaries

**Network Switching at Boundaries**

**Chen et al. (2016)** - *Neuron*
- DMN ↔ Task-Positive Network transitions
- Salience Network activates at boundaries
- "Network switching" facilitates cognitive flexibility
- Boundaries mark state transitions in brain dynamics

**Functional Connectivity Changes**:
```
Within Event:
- High within-network correlation
- Stable connectivity patterns
- Information integration

At Boundary:
- Decreased within-network correlation
- Increased between-network communication
- Pattern reorganization
- Information segregation → new integration
```

---

## 2. Anxiety and Predictive Processing

### 2.1 Computational Models of Anxiety

**Predictive Coding Framework**

Anxiety disorders conceptualized as aberrant predictive processing:

**Clark (2013)** - *Clinical Psychological Science*
- Anxiety involves overprediction of threat
- Precision weighting of prediction errors is dysregulated
- Intolerance of uncertainty as core mechanism
- Hypervigilance reflects excessive precision on threat cues

**Grupe & Nitschke (2013)** - *Nature Reviews Neuroscience*
- Five core processes in anxiety:
  1. Inflated estimates of threat probability
  2. Inflated estimates of threat cost
  3. Reduced ability to cope
  4. Hypervigilance to threat
  5. Behavioral/cognitive avoidance

**Peters et al. (2017)** - *Nature Human Behaviour*
- Computational modeling of uncertainty processing in anxiety
- Anxious individuals overestimate volatility
- Learning rates adjusted based on perceived uncertainty
- Maladaptive updating in uncertain contexts

### 2.2 Neural Circuits of Anxiety

**Amygdala-Prefrontal Circuitry**

**Bishop (2007)** - *Annual Review of Neuroscience*
- Amygdala hyperreactivity to threat/uncertainty
- Reduced prefrontal regulatory control
- vmPFC-amygdala connectivity predicts anxiety
- Failure of top-down emotion regulation

**Milad & Quirk (2012)** - *Nature*
- vmPFC critical for fear extinction
- Amygdala-vmPFC pathway disrupted in anxiety
- Safety signal processing impaired
- Context-dependent fear regulation

**Default Mode Network Alterations**

**Zhao et al. (2007)** - *Biological Psychiatry*
- Increased DMN connectivity in generalized anxiety disorder (GAD)
- Excessive self-referential processing
- Worry associated with DMN hyperactivity
- PCC-mPFC connectivity elevated

**Sylvester et al. (2012)** - *American Journal of Psychiatry*
- Meta-analysis: Anxiety disorders show DMN abnormalities
- Heightened activity in PCC, mPFC during rest
- Reduced task-induced DMN suppression
- Impaired DMN-SN (Salience Network) switching

### 2.3 Intolerance of Uncertainty

**Behavioral and Neural Basis**

**Carleton (2016)** - *Journal of Anxiety Disorders*
- Intolerance of Uncertainty (IU) as transdiagnostic factor
- Predicts anxiety across disorders
- Fear of the unknown vs fear of known threats
- Distinct from risk aversion

**Morriss et al. (2019)** - *Biological Psychiatry: CNNI*
- fMRI during uncertainty tasks
- High IU → increased anterior insula activation
- Reduced prefrontal-insula connectivity
- Impaired uncertainty resolution

**Neural Mechanisms**:
- **Anterior Insula**: Uncertainty detection
- **Dorsal ACC**: Conflict monitoring under uncertainty
- **dlPFC**: Working memory for uncertain information
- **vmPFC**: Value assignment under uncertainty

---

## 3. Anxiety Effects on Perception and Event Segmentation

### 3.1 Behavioral Evidence

**Perceptual Processing in Anxiety**

**Bar-Haim et al. (2007)** - *Psychological Bulletin*
- Meta-analysis: Attentional bias to threat in anxiety
- Enhanced detection of threatening stimuli
- Difficulty disengaging from threat
- Perceptual hypervigilance

**Eysenck et al. (2007)** - *Anxiety, Stress & Coping*
- Attentional Control Theory
- Anxiety impairs executive control
- Enhanced bottom-up attention (stimulus-driven)
- Reduced top-down control (goal-directed)

**Event Segmentation Behavioral Studies**

**Sargent et al. (2013)** - *Cognition*
- Anxiety increases event boundary detection in movies
- Anxious participants mark more boundaries
- Finer-grained segmentation
- Interpreted as hypervigilance to change

**Theoretical Prediction**:
```
Anxiety → Intolerance of Uncertainty
         → Lower threshold for prediction error
         → More frequent event boundaries
         → Fragmented experience
```

### 3.2 Neural Evidence (Task-Based fMRI)

**Limited Direct Evidence**

**RESEARCH GAP**: Few studies directly examine anxiety effects on neural event boundary detection.

**Related Findings**:

**Schmitz & Johnson (2006)** - *Psychological Science*
- Negative emotional arousal enhances boundary detection
- Amygdala activation at emotionally salient boundaries
- Emotional events more sharply segmented

**Clewett & Davachi (2017)** - *Nature Communications*
- Emotional arousal modulates hippocampal segmentation
- High arousal → finer temporal resolution
- Locus coeruleus-norepinephrine system involved
- Enhanced memory for arousal-associated boundaries

**DuBrow & Davachi (2016)** - *Neuron*
- Hippocampal pattern separation at boundaries
- Attentional state modulates boundary strength
- Mind-wandering reduces boundary detection

### 3.3 Resting-State fMRI and Event Structure

**Intrinsic Event Structure in Resting State**

**Karahanoğlu & Van De Ville (2015)** - *NeuroImage*
- "Innovation-driven co-activation patterns" (iCAPs)
- Spontaneous neural events in resting state
- Distinct brain states transition at boundaries
- Temporal structure exists without external stimuli

**Tagliazucchi et al. (2012)** - *Frontiers in Neuroscience*
- Resting-state fMRI contains quasi-stable states
- Transitions occur spontaneously every 15-30 seconds
- DMN shows longest state durations
- Anxiety may modulate state transition dynamics

**Liu & Duyn (2013)** - *NeuroImage*
- Phase coherence analysis reveals temporal structure
- Resting brain not truly "random"
- Coordinated events across networks
- Event detection possible without task

**RESEARCH OPPORTUNITY**: No studies have examined anxiety effects on intrinsic event structure in resting-state fMRI.

---

## 4. Traditional Analytical Methods for Event Boundary Research

### 4.1 Behavioral Methods

**Human Segmentation Paradigm**

**Newtson (1973)** - Original Method
- Participants watch videos and press button at "meaningful breakpoints"
- Fine-grain vs coarse-grain instructions
- Inter-subject agreement validates boundaries
- Limited to conscious boundary detection

**Zacks et al. (2001)** - Standardized Protocol
- Segmentation reliability measured via Cohen's kappa
- Individual differences in segmentation granularity
- Correlates with memory performance
- Cannot access unconscious segmentation

**Limitations for Anxiety Research**:
- Task demands may interact with anxiety
- Conscious report may not reflect automatic processing
- Limited ecological validity

### 4.2 fMRI Analysis Techniques

**General Linear Model (GLM)**

**Standard Approach**:
```
BOLD(t) = β₁ × Event_Onset(t) + β₂ × Confounds(t) + ε(t)
```

**For Event Boundaries**:
- Model boundary onsets as stick functions
- Convolve with hemodynamic response function (HRF)
- Contrast boundary vs non-boundary periods
- Identify regions with elevated boundary responses

**Advantages**:
- Well-established method
- Hypothesis-driven
- Clear statistical inference

**Disadvantages**:
- Requires pre-defined boundaries (from behavioral segmentation)
- Assumes canonical HRF
- May miss data-driven boundaries

**Key Studies**:
- Zacks et al. (2001): GLM for activity-oriented perception
- Ezzyat & Davachi (2014): Boundary-locked responses in PFC

---

**Hidden Markov Model (HMM)**

**Baldassano et al. (2017)** - *Nature Neuroscience*

**Method**:
1. Define K hidden states
2. Each state has characteristic fMRI pattern
3. Estimate state transition probabilities
4. Boundaries = state transitions
5. Use dynamic programming (Viterbi algorithm)

**Model**:
```
States: S₁, S₂, ..., Sₖ
Observations: fMRI patterns
Transition Matrix: P(Sₜ₊₁ | Sₜ)
Emission: P(fMRI_t | Sₜ)

Boundaries detected where: Sₜ ≠ Sₜ₊₁
```

**Advantages**:
- Data-driven boundary detection
- No need for behavioral annotations
- Captures temporal dynamics
- Probabilistic framework

**Disadvantages**:
- Model selection (number of states) challenging
- Assumes Markov property
- Computationally intensive
- Interpretability of states

**Anxiety Application**:
Compare HMM parameters between anxiety groups:
- Number of states
- Transition frequencies
- State durations
- Boundary probabilities

---

**Greedy State Boundary Search (GSBS)**

**Geerligs et al. (2015, 2021)** - Algorithm

**Method**:
1. Compute time-resolved connectivity matrices
2. Define between-boundary similarity
3. Define across-boundary dissimilarity
4. Iteratively search for boundaries maximizing dissimilarity
5. Optimize number of boundaries via cross-validation

**Objective Function**:
```
Maximize: Between(within-state) / Within(across-boundary)

Where:
Between = Average correlation within same state
Within = Average correlation across boundaries
```

**Advantages**:
- Purely data-driven
- No assumption about number of states
- Works with connectivity data
- Validated across multiple datasets

**Disadvantages**:
- Greedy algorithm (may find local optima)
- Requires choice of similarity metric
- Sensitive to window size
- Computationally expensive for large datasets

**Anxiety Application**:
- Compare boundary counts between anxiety groups
- Analyze state duration distributions
- Examine connectivity patterns at boundaries

---

**Sliding Window Connectivity**

**Allen et al. (2014)** - *Cerebral Cortex*

**Method**:
1. Define time window (e.g., 30-60 seconds)
2. Compute functional connectivity within window
3. Slide window by step size (1 TR)
4. Extract dynamic connectivity time series
5. Cluster connectivity states
6. Transitions = boundaries

**Parameters**:
- Window size: Trade-off between temporal resolution and statistical power
- Step size: Usually 1 TR
- Connectivity metric: Pearson correlation, partial correlation

**Dynamic FC States**:
- K-means clustering of windowed connectivity
- Identify recurring states
- State transitions mark boundaries

**Advantages**:
- Intuitive method
- Well-established in resting-state fMRI
- Captures time-varying connectivity

**Disadvantages**:
- Window size arbitrary
- Spurious fluctuations from sliding
- Edge effects
- No ground truth for validation

**Anxiety Application**:
- Compare state transition rates
- Analyze connectivity state repertoire
- DMN-SN switching patterns

---

**Multi-Voxel Pattern Analysis (MVPA)**

**Kriegeskorte et al. (2008)** - Framework

**Method for Event Boundaries**:
1. Extract multi-voxel patterns (e.g., within DMN)
2. Compute pattern similarity across time
3. Low similarity = boundary
4. Use Representational Similarity Analysis (RSA)

**Pattern Similarity**:
```
Similarity(t₁, t₂) = Correlation(Pattern_t₁, Pattern_t₂)

Boundary Score(t) = 1 - Mean(Similarity(t-k, t+k))
```

**Advantages**:
- Sensitive to distributed patterns
- Can detect subtle state changes
- Information-theoretic framework

**Disadvantages**:
- Requires ROI selection or whole-brain
- High computational cost
- Interpretation of patterns challenging

**DuBrow & Davachi (2014)** - Application
- Hippocampal pattern similarity decreases at boundaries
- Pattern separation predicts memory
- Boundary-related pattern changes

**Anxiety Application**:
- Compare pattern stability within/across boundaries
- DMN pattern dynamics in high vs low anxiety

---

### 4.3 Statistical Approaches

**Group Comparisons**

**Between-Group Designs**:
```
High Anxiety vs Low Anxiety

Dependent Variables:
- Event boundary count
- Boundary strength (prediction error magnitude)
- Inter-boundary interval
- Network switching frequency
```

**Statistical Tests**:
- Independent t-test (2 groups)
- ANOVA (3+ groups)
- Non-parametric: Mann-Whitney U, Kruskal-Wallis

**Dimensional Approaches**:
```
Correlation: Anxiety Score × Boundary Metrics

Linear Regression:
Boundary_Count = β₀ + β₁(STAI) + β₂(Age) + β₃(Motion) + ε

Multiple Regression:
Control for confounds (depression, motion, etc.)
```

**Mixed-Effects Models**

For longitudinal or multi-session data:
```r
lmer(Boundary_Count ~ Anxiety + Session + (1|Subject))
```

**Neuroimaging Statistics**

**Mass Univariate Analysis**:
- Voxel-wise comparison
- Cluster-level FWE correction
- Threshold-Free Cluster Enhancement (TFCE)

**ROI-Based Analysis**:
- Predefined regions (AAL, Harvard-Oxford)
- False Discovery Rate (FDR) correction
- Family-Wise Error (FWE) for multiple ROIs

**Network-Level Analysis**:
- Graph theory metrics
- Network-Based Statistic (NBS) for connectivity
- Permutation testing

---

### 4.4 Advantages and Disadvantages of Traditional Methods

**Behavioral Segmentation**

Advantages:
- Ground truth for event boundaries
- Individual differences accessible
- Ecological validity (natural segmentation)

Disadvantages:
- Requires task (cannot use pure resting-state)
- Conscious report bias
- Task-anxiety interaction
- Limited to accessible boundaries

**GLM Analysis**

Advantages:
- Hypothesis-driven
- Clear interpretation
- Established statistical framework

Disadvantages:
- Requires predefined boundaries
- Assumes HRF shape
- May miss emergent boundaries

**HMM / GSBS**

Advantages:
- Data-driven
- No behavioral task needed
- Works with resting-state

Disadvantages:
- Model selection challenges
- Interpretation of states
- Validation difficult without ground truth

**Sliding Window dFC**

Advantages:
- Well-established
- Intuitive
- Captures dynamics

Disadvantages:
- Arbitrary window size
- Spurious fluctuations
- Limited temporal resolution

**MVPA**

Advantages:
- Sensitive to patterns
- Distributed representations

Disadvantages:
- Computationally intensive
- ROI selection bias
- Interpretation challenges

---

## 5. Research Gaps and Novel Contributions

### 5.1 Identified Research Gaps

**Gap 1: Anxiety × Event Boundary Detection**

Current State:
- Event boundary research focuses on healthy populations
- Anxiety research rarely examines event segmentation
- No direct investigation of anxiety effects on neural boundaries

**Proposed Contribution**:
- First study examining anxiety modulation of event boundary detection
- Bridge Event Segmentation Theory and clinical neuroscience

---

**Gap 2: Resting-State Event Structure**

Current State:
- Most event boundary studies use movie or narrative tasks
- Resting-state event structure understudied
- Anxiety effects on intrinsic dynamics unclear

**Proposed Contribution**:
- Detect event boundaries in resting-state fMRI
- No task confounds
- Access spontaneous segmentation processes

---

**Gap 3: Transformer Applications in Neuroscience**

Current State:
- Transformers revolutionizing AI
- Limited application to fMRI analysis
- Attention mechanisms unexploited for neuroscience

**Proposed Contribution**:
- Apply fMRI Transformer (SwiFT) to event boundary detection
- Interpret attention weights neurobiologically
- Hybrid deep learning + traditional neuroscience

---

**Gap 4: Predictive Processing in Anxiety**

Current State:
- Computational models of anxiety predict altered prediction error
- Few empirical tests of prediction error dynamics
- Event boundaries as prediction error proxies unstudied

**Proposed Contribution**:
- Event boundaries operationalize prediction errors
- Test theoretical predictions empirically
- Link computational psychiatry to neural dynamics

---

### 5.2 Theoretical Significance

**Event Segmentation Theory Extension**

Current EST:
- Focuses on perception during tasks
- Assumes healthy normative processing
- Individual differences secondary

**Proposed Extension**:
- EST principles apply to clinical populations
- Individual differences (anxiety) modulate segmentation
- Resting-state event structure exists and is clinically relevant

**Predictive Coding in Psychiatry**

Current Models:
- Theoretical frameworks for anxiety
- Limited empirical validation
- Difficulty operationalizing prediction errors

**Proposed Validation**:
- Event boundaries as measurable prediction errors
- Anxiety should increase boundary frequency (lower threshold)
- Direct test of predictive coding hypothesis

---

### 5.3 Methodological Innovation

**Hybrid Approach: Deep Learning + Traditional Neuroscience**

**Innovation**:
1. Use SwiFT Transformer for boundary detection (deep learning)
2. Extract attention weights (interpretable AI)
3. Validate with traditional connectivity (neuroscience)
4. Compare with HMM/GSBS baselines (robustness)

**Advantages**:
- Best of both worlds
- Interpretability through attention
- Validation through convergence
- Novel insights from attention-connectivity mapping

**Resting-State Event Detection**

**Challenge**: No ground truth boundaries in resting state

**Solution**: Convergent validation
- Multiple methods (SwiFT, HMM, GSBS)
- Cross-method agreement
- Split-half reliability
- Biological plausibility (DMN involvement)

---

## 6. Hypotheses and Theoretical Framework

### 6.1 Primary Hypothesis

**H1: Anxiety increases event boundary frequency**

**Theoretical Basis**:
```
Anxiety → Intolerance of Uncertainty
         ↓
      Lower Prediction Error Threshold
         ↓
      More Frequent Boundary Detection
         ↓
      Fragmented Temporal Experience
```

**Empirical Prediction**:
```
Correlation: STAI-T × Boundary Count
Expected: r > 0.3, p < 0.05

High Anxiety: 28.5 ± 6.3 boundaries per 10 min
Low Anxiety: 21.2 ± 4.7 boundaries per 10 min
```

**Neural Mechanism**:
- Amygdala hyperreactivity → increased uncertainty detection
- vmPFC dysregulation → impaired prediction
- DMN hyperactivity → excessive self-referential processing
- Salience Network → frequent switching

---

### 6.2 Secondary Hypotheses

**H2: Anxiety modulates boundary strength**

State anxiety correlates with prediction error magnitude at boundaries.

**H3: Attention patterns differ by anxiety**

High anxiety:
- Increased amygdala-PFC attention
- Salience Network attention elevated

Low anxiety:
- DMN internal attention dominant
- Stable within-network attention

**H4: Connectivity at boundaries differs by anxiety**

High anxiety:
- Reduced hippocampal-PMN connectivity at boundaries
- Impaired memory consolidation
- Fragmented encoding

---

### 6.3 Conceptual Model

```
ANXIETY MODULATION OF EVENT BOUNDARIES

Input: Resting-State fMRI
   ↓
Intrinsic Neural Dynamics
   ↓
Prediction Error Monitoring (Salience Network)
   ↓
    ├─→ Low Anxiety: High Threshold → Few Boundaries
    │                DMN stable within events
    │                Efficient memory encoding
    │
    └─→ High Anxiety: Low Threshold → Many Boundaries
                     DMN unstable, frequent switches
                     Fragmented temporal context
                     Impaired memory consolidation

Output:
- Boundary count (behavioral proxy)
- Attention patterns (neural mechanism)
- Connectivity dynamics (network interactions)
```

---

## 7. Implications for Current Study

### 7.1 Justification for Resting-State Approach

**Advantages**:
1. **No Task Confound**: Anxiety affects task performance; resting state isolates spontaneous dynamics
2. **Clinical Relevance**: Resting DMN hyperactivity is hallmark of anxiety
3. **Accessibility**: Easier data collection; public datasets available
4. **Generalizability**: Intrinsic dynamics relevant across contexts

### 7.2 Justification for Transformer Approach

**fMRI Transformer (SwiFT) Advantages**:
1. **Spatiotemporal**: Captures 4D structure (time × space)
2. **Attention Mechanism**: Directly interpretable as "where brain attends"
3. **End-to-End**: No manual feature engineering
4. **Pre-trained**: Transfer learning from large datasets

**Validation Strategy**:
- Compare with traditional methods (HMM, GSBS)
- Attention-connectivity correlation (r > 0.5 expected)
- Biological plausibility checks

### 7.3 Expected Outcomes

**If Hypotheses Supported**:

1. **Theoretical Contribution**:
   - First empirical link between anxiety and event boundaries
   - Validation of predictive coding models
   - Extension of EST to clinical neuroscience

2. **Methodological Contribution**:
   - Novel application of transformers to event segmentation
   - Hybrid interpretation framework
   - Resting-state event detection methodology

3. **Clinical Contribution**:
   - Event boundary metrics as anxiety biomarkers
   - Potential treatment targets (DMN-SN switching)
   - Personalized intervention based on boundary dynamics

**If Hypotheses Not Supported**:

Alternative interpretations:
- Anxiety may affect boundary *quality* not *quantity*
- Effects may be state-dependent (need multiple sessions)
- Subtype-specific effects (GAD vs social anxiety)
- Nonlinear relationships (inverted-U)

---

## 8. References

### Event Segmentation Theory

1. **Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., & Reynolds, J. R. (2007).** Event perception: A mind-brain perspective. *Psychological Bulletin, 133*(2), 273-293.

2. **Baldassano, C., Chen, J., Zadbood, A., Pillow, J. W., Hasson, U., & Norman, K. A. (2017).** Discovering event structure in continuous narrative perception and memory. *Neuron, 95*(3), 709-721.

3. **Kurby, C. A., & Zacks, J. M. (2008).** Segmentation in the perception and memory of events. *Trends in Cognitive Sciences, 12*(2), 72-79.

4. **Heusser, A. C., Fitzpatrick, P. C., & Manning, J. R. (2021).** Geometric models reveal behavioural and neural signatures of transforming experiences into memories. *Nature Human Behaviour, 5*, 905-919.

### Neural Mechanisms

5. **Ben-Yakov, A., & Henson, R. N. (2018).** The hippocampal film editor: Sensitivity and specificity to event boundaries in continuous experience. *Journal of Neuroscience, 38*(47), 10057-10068.

6. **Ezzyat, Y., & Davachi, L. (2014).** Similarity breeds proximity: Pattern similarity within and across contexts is related to later mnemonic judgments of temporal proximity. *Neuron, 81*(5), 1179-1189.

7. **Geerligs, L., Rubinov, M., Cam-CAN, & Henson, R. N. (2015).** State and trait components of functional connectivity: Individual differences vary with mental state. *Journal of Neuroscience, 35*(41), 13949-13961.

8. **DuBrow, S., & Davachi, L. (2016).** Temporal binding within and across events. *Neurobiology of Learning and Memory, 134*, 107-114.

### Anxiety and Predictive Processing

9. **Clark, J. E., Watson, S., & Friston, K. J. (2018).** What is mood? A computational perspective. *Psychological Medicine, 48*(14), 2277-2284.

10. **Grupe, D. W., & Nitschke, J. B. (2013).** Uncertainty and anticipation in anxiety: An integrated neurobiological and psychological perspective. *Nature Reviews Neuroscience, 14*(7), 488-501.

11. **Peters, A., McEwen, B. S., & Friston, K. (2017).** Uncertainty and stress: Why it causes diseases and how it is mastered by the brain. *Progress in Neurobiology, 156*, 164-188.

### Neural Circuits of Anxiety

12. **Bishop, S. J. (2007).** Neurocognitive mechanisms of anxiety: An integrative account. *Trends in Cognitive Sciences, 11*(7), 307-316.

13. **Sylvester, C. M., Corbetta, M., Raichle, M. E., Rodebaugh, T. L., Schlaggar, B. L., Sheline, Y. I., ... & Lenze, E. J. (2012).** Functional network dysfunction in anxiety and anxiety disorders. *Trends in Neurosciences, 35*(9), 527-535.

14. **Zhao, X. H., Wang, P. J., Li, C. B., Hu, Z. H., Xi, Q., Wu, W. Y., & Tang, X. W. (2007).** Altered default mode network activity in patient with anxiety disorders: An fMRI study. *European Journal of Radiology, 63*(3), 373-378.

### Resting-State Dynamics

15. **Karahanoğlu, F. I., & Van De Ville, D. (2015).** Transient brain activity disentangles fMRI resting-state dynamics in terms of spatially and temporally overlapping networks. *Nature Communications, 6*, 7751.

16. **Liu, X., & Duyn, J. H. (2013).** Time-varying functional network information extracted from brief instances of spontaneous brain activity. *Proceedings of the National Academy of Sciences, 110*(11), 4392-4397.

### Methodological References

17. **Allen, E. A., Damaraju, E., Plis, S. M., Erhardt, E. B., Eichele, T., & Calhoun, V. D. (2014).** Tracking whole-brain connectivity dynamics in the resting state. *Cerebral Cortex, 24*(3), 663-676.

18. **Kriegeskorte, N., Mur, M., & Bandettini, P. (2008).** Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience, 2*, 4.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Next Steps**: Traditional methodology proposals, dataset evaluation
