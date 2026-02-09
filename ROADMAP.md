# Eddy-Current Inverse Problem: Project Roadmap

## 🎯 Project Goal

**Build a complete forward-inverse eddy-current workflow** that can:
1. Generate material profiles (σ, μ) using GANs and analytical methods
2. Calculate eddy-current response (EDC) from material profiles (Forward Problem)
3. Recover material profiles from measured EDC (Inverse Problem)
4. Match recovered profiles against a database of known materials

**Ultimate Objective**: Given only EDC measurements from a real metal sample, identify its material properties (σ, μ profiles).

---

## 📋 Project Phases

### Phase 1: Foundation & Infrastructure (Weeks 1-2)

**Goal**: Establish core utilities and configuration management

**What We Have**:
- ✅ Profile generation (Roberts plan, 4 profile types)
- ✅ Discretization (continuous → K layers)
- ✅ Trained GANs (Dual WGAN, Improved WGAN v2)

**What We Need**:
- [ ] `ProfileNormalizer` class for GAN compatibility
- [ ] `GlobalConfig` - single source of truth for all parameters
- [ ] Package structure setup
- [ ] Unit tests for normalization

**Implementation Items**:
1. Create `eddy_current_workflow/` package structure
2. Implement `config/global_config.py`
3. Implement `profiles/normalization.py`
4. Write tests for round-trip normalization
5. Document all configuration parameters

**Success Criteria**:
- Normalization is lossless (< 1e-10 error)
- All parameters centralized in GlobalConfig
- Package imports work correctly

---

### Phase 2: Forward EDC Model (Weeks 3-4)

**Goal**: Implement physics-based forward solver (σ, μ) → EDC

**What We Have**:
- ✅ Discretized profiles (K=51 layers)
- ✅ Material parameter bounds

**What We Need**:
- [ ] EDC physics equations from monograph
- [ ] `ProbeSettings` dataclass
- [ ] `EDCResponse` dataclass
- [ ] `edc_forward()` function with actual physics
- [ ] Validation against analytical solutions

**Implementation Items**:
1. Extract EDC equations from Chapter 2/3 of monograph
2. Define probe geometry and excitation parameters
3. Implement forward solver:
   - Option A: Analytical solution (Dodd-Deeds for layered media)
   - Option B: Numerical solver (FEM/BEM)
   - Option C: Surrogate model (if physics solver too slow)
4. Create test cases with known solutions
5. Validate numerical accuracy and stability

**Success Criteria**:
- Forward solver produces consistent EDC for same input
- Matches analytical solutions (if available)
- Runs in reasonable time (< 1 second per profile)
- Handles edge cases (uniform layers, extreme values)

---

### Phase 3: Inverse Problem Solver (Weeks 5-6)

**Goal**: Recover (σ, μ) from EDC measurements

**What We Have**:
- ✅ Forward solver (from Phase 2)
- ✅ Profile bounds and constraints

**What We Need**:
- [ ] `EDCMismatchObjective` class
- [ ] `InverseSolver` with multiple strategies
- [ ] Regularization options (smoothness, monotonicity)
- [ ] Constraint handling

**Implementation Items**:
1. Define mismatch functional Z(EDC_gen - EDC_measured)
2. Implement objective function with gradients (if possible)
3. Implement optimization strategies:
   - Multi-start local optimization (L-BFGS-B)
   - Global optimization (Differential Evolution)
   - Optional: CMA-ES, Bayesian optimization
4. Add regularization penalties
5. Handle physical constraints (σ > 0, μ > 0)

**Success Criteria**:
- Recovers known profiles from synthetic EDC (< 5% error)
- Handles noisy measurements robustly
- Converges in reasonable time (< 5 minutes)
- Provides uncertainty estimates

---

### Phase 4: Profile Database & Search (Weeks 7-8)

**Goal**: Store and retrieve material profiles efficiently

**What We Have**:
- ✅ Generated training profiles (2000 samples)
- ✅ GAN-generated profiles

**What We Need**:
- [ ] `ProfileDatabase` with HDF5 storage
- [ ] `ProfileSearchEngine` for nearest-neighbor search
- [ ] Distance metrics in profile space
- [ ] Metadata storage

**Implementation Items**:
1. Design database schema (HDF5 format)
2. Implement storage operations (add, retrieve, update)
3. Implement search algorithms:
   - Exact k-NN search
   - Approximate NN (if database large)
   - Multiple distance metrics (Euclidean, cosine, correlation)
4. Add metadata (profile type, parameters, EDC if pre-computed)
5. Populate database with existing profiles

**Success Criteria**:
- Fast retrieval (< 100ms for k-NN on 10k profiles)
- Accurate matching (finds similar profiles)
- Scalable to 100k+ profiles
- Supports incremental updates

---

### Phase 5: End-to-End Integration (Weeks 9-10)

**Goal**: Connect all components into working pipelines

**What We Have**:
- ✅ All individual components (from Phases 1-4)

**What We Need**:
- [ ] `forward_pipeline.py` - Profile → EDC
- [ ] `inverse_pipeline.py` - EDC → Profile → DB match
- [ ] `validation_pipeline.py` - Round-trip testing
- [ ] Integration tests

**Implementation Items**:
1. Implement forward pipeline with all profile sources:
   - Analytical profiles (Roberts plan)
   - GAN-generated profiles
   - Manual profiles
2. Implement inverse pipeline:
   - EDC input → optimization → recovered profile
   - Database search → nearest matches
   - Confidence metrics
3. Implement validation pipeline:
   - Synthetic data generation
   - Round-trip error analysis
   - Noise sensitivity testing
4. Create example scripts and tutorials

**Success Criteria**:
- All pipelines work end-to-end
- Clear error messages and logging
- Documented examples
- Performance benchmarks

---

### Phase 6: Validation & Optimization (Weeks 11-12)

**Goal**: Validate on real/realistic data and optimize performance

**What We Have**:
- ✅ Complete pipeline (from Phase 5)

**What We Need**:
- [ ] Validation datasets (synthetic + real if available)
- [ ] Performance benchmarks
- [ ] Error analysis
- [ ] Documentation

**Implementation Items**:
1. Create comprehensive test suite:
   - Unit tests for each component
   - Integration tests for pipelines
   - Regression tests
2. Validate on synthetic data:
   - Known profiles → EDC → recovered profiles
   - Various noise levels
   - Different profile types
3. Performance optimization:
   - Profile forward solver bottlenecks
   - Optimize inverse solver convergence
   - Parallelize where possible
4. Document results and limitations

**Success Criteria**:
- < 5% error on noise-free synthetic data
- < 10% error with 5% noise
- Complete documentation
- Published results/report

---

## 🔧 Technical Requirements

### Software Stack
- **Python**: 3.8+
- **Core Libraries**: NumPy, SciPy, PyTorch
- **Storage**: HDF5 (h5py)
- **Optimization**: scipy.optimize, optional CMA-ES
- **Visualization**: Matplotlib
- **Testing**: pytest

### Hardware Requirements
- **CPU**: Multi-core recommended for optimization
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (for GAN generation)
- **Storage**: 10GB for database

### External Dependencies
- **Physics Solver**: TBD based on monograph analysis
  - Option A: Analytical (implement from equations)
  - Option B: FEM library (FEniCS, COMSOL API)
  - Option C: Pre-computed LUT with interpolation

---

## 📊 Success Metrics

### Technical Metrics
- **Forward Solver Accuracy**: Match analytical solutions within 0.1%
- **Inverse Recovery Error**: < 5% on synthetic data
- **Noise Robustness**: < 10% error with 5% measurement noise
- **Speed**: Forward < 1s, Inverse < 5 min per sample
- **Database Search**: < 100ms for k-NN

### Deliverables
- [ ] Working forward-inverse pipeline
- [ ] Profile database with 10k+ samples
- [ ] Comprehensive test suite (> 80% coverage)
- [ ] Documentation (API + user guide)
- [ ] Validation report with results
- [ ] Example notebooks/scripts

---

## 🚨 Critical Unknowns & Risks

### High Priority
1. **EDC Physics Equations**: Must extract from monograph
   - Risk: Equations may be complex/numerical
   - Mitigation: Start with simplified model, iterate

2. **Forward Solver Performance**: May be too slow
   - Risk: Inverse optimization requires many forward calls
   - Mitigation: Pre-compute LUT, use surrogate model

3. **Inverse Problem Uniqueness**: Multiple (σ, μ) may give same EDC
   - Risk: Non-unique solutions
   - Mitigation: Regularization, prior knowledge, database matching

### Medium Priority
4. **Real Data Availability**: May not have validation data
   - Risk: Cannot validate on real measurements
   - Mitigation: Extensive synthetic validation

5. **Computational Resources**: Optimization may be expensive
   - Risk: Long computation times
   - Mitigation: Parallel optimization, GPU acceleration

---

## 📅 Timeline Summary

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| Phase 1 | Weeks 1-2 | Foundation ready |
| Phase 2 | Weeks 3-4 | Forward solver working |
| Phase 3 | Weeks 5-6 | Inverse solver working |
| Phase 4 | Weeks 7-8 | Database operational |
| Phase 5 | Weeks 9-10 | End-to-end pipeline |
| Phase 6 | Weeks 11-12 | Validated & documented |

**Total Duration**: ~12 weeks (3 months)

---

## 🎓 Learning Resources

### Eddy-Current Theory
- Monograph Chapter 2: Material models and discretization
- Monograph Chapter 3: EDC response equations (TBD)
- Dodd-Deeds papers on layered media

### Inverse Problems
- Tarantola: Inverse Problem Theory
- Kaipio & Somersalo: Statistical and Computational Inverse Problems

### Optimization
- Nocedal & Wright: Numerical Optimization
- CMA-ES documentation

---

## 📝 Current Status

**Phase**: Phase 1 (Foundation)
**Progress**: 0% complete
**Next Action**: 
1. Create package structure
2. Implement GlobalConfig
3. Implement ProfileNormalizer
4. Search monograph for EDC equations

**Last Updated**: February 1, 2026

---

**END OF ROADMAP**
