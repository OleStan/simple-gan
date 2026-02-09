#!/usr/bin/env python
"""Test Phase 1: Foundation components."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eddy_current_workflow.config import GlobalConfig, CONFIG
from eddy_current_workflow.profiles import ProfileNormalizer
from eddy_current_workflow.forward import ProbeSettings, EDCResponse, edc_forward


def test_global_config():
    """Test GlobalConfig functionality."""
    print("="*70)
    print("Testing GlobalConfig")
    print("="*70)
    
    config = GlobalConfig()
    
    print(f"\n{config}")
    
    print(f"\nLayer thickness: {config.layer_thickness*1e6:.2f} μm")
    print(f"Sigma range: [{config.sigma_bounds[0]:.2e}, {config.sigma_bounds[1]:.2e}] S/m")
    print(f"Mu range: [{config.mu_bounds[0]:.2f}, {config.mu_bounds[1]:.2f}]")
    
    config.validate()
    print("✓ Configuration validation passed")
    
    config.save('./test_config.json')
    print("✓ Configuration saved to test_config.json")
    
    config_loaded = GlobalConfig.load('./test_config.json')
    print("✓ Configuration loaded from file")
    
    assert config_loaded.K == config.K
    assert config_loaded.sigma_bounds == config.sigma_bounds
    print("✓ Loaded configuration matches original")
    
    print("\n" + "="*70)
    print("GlobalConfig: PASSED ✓")
    print("="*70 + "\n")


def test_profile_normalizer():
    """Test ProfileNormalizer functionality."""
    print("="*70)
    print("Testing ProfileNormalizer")
    print("="*70)
    
    K = 51
    sigma = np.linspace(1e6, 6e7, K)
    mu = np.linspace(1.0, 100.0, K)
    
    normalizer = ProfileNormalizer(
        sigma_min=1e6,
        sigma_max=6e7,
        mu_min=1.0,
        mu_max=100.0
    )
    
    print(f"\n{normalizer}")
    
    sigma_norm, mu_norm = normalizer.normalize(sigma, mu)
    print(f"\nNormalized sigma range: [{sigma_norm.min():.4f}, {sigma_norm.max():.4f}]")
    print(f"Normalized mu range: [{mu_norm.min():.4f}, {mu_norm.max():.4f}]")
    
    assert np.allclose(sigma_norm.min(), -1.0, atol=1e-10)
    assert np.allclose(sigma_norm.max(), 1.0, atol=1e-10)
    assert np.allclose(mu_norm.min(), -1.0, atol=1e-10)
    assert np.allclose(mu_norm.max(), 1.0, atol=1e-10)
    print("✓ Normalization to [-1, 1] correct")
    
    sigma_recovered, mu_recovered = normalizer.denormalize(sigma_norm, mu_norm)
    
    is_valid, errors = normalizer.validate_roundtrip(sigma, mu)
    print(f"\nRound-trip validation:")
    print(f"  Sigma max error: {errors['sigma_max_error']:.2e}")
    print(f"  Mu max error: {errors['mu_max_error']:.2e}")
    print(f"  Sigma relative error: {errors['sigma_rel_error']:.2e}")
    print(f"  Mu relative error: {errors['mu_rel_error']:.2e}")
    
    assert is_valid, "Round-trip validation failed"
    print("✓ Round-trip normalization is lossless")
    
    normalizer.save('./test_normalizer.json')
    print("✓ Normalizer saved to test_normalizer.json")
    
    normalizer_loaded = ProfileNormalizer.load('./test_normalizer.json')
    print("✓ Normalizer loaded from file")
    
    sigma_norm2, mu_norm2 = normalizer_loaded.normalize(sigma, mu)
    assert np.allclose(sigma_norm, sigma_norm2)
    assert np.allclose(mu_norm, mu_norm2)
    print("✓ Loaded normalizer produces identical results")
    
    sigma_data = np.random.uniform(1e6, 6e7, (100, K))
    mu_data = np.random.uniform(1.0, 100.0, (100, K))
    normalizer_auto = ProfileNormalizer.from_data(sigma_data, mu_data, margin=0.05)
    print(f"\n✓ Created normalizer from data with 5% margin")
    print(f"  Auto sigma bounds: [{normalizer_auto.sigma_min:.2e}, {normalizer_auto.sigma_max:.2e}]")
    print(f"  Auto mu bounds: [{normalizer_auto.mu_min:.2f}, {normalizer_auto.mu_max:.2f}]")
    
    print("\n" + "="*70)
    print("ProfileNormalizer: PASSED ✓")
    print("="*70 + "\n")


def test_forward_edc():
    """Test forward EDC solver (Dodd-Deeds)."""
    print("="*70)
    print("Testing Forward EDC Solver (Dodd-Deeds)")
    print("="*70)
    
    K = 51
    sigma_layers = np.linspace(1.88e7, 3.766e7, K)
    mu_layers = np.linspace(8.8, 1.0, K)
    
    settings = ProbeSettings(
        frequency=1e6,
        inner_radius=4e-3,
        outer_radius=6e-3,
        lift_off=0.5e-3,
    )
    
    print(f"\nProbe settings:")
    print(f"  Frequency: {settings.frequency/1e6:.2f} MHz")
    print(f"  Coil radii: [{settings.inner_radius*1e3:.1f}, {settings.outer_radius*1e3:.1f}] mm")
    print(f"  Lift-off: {settings.lift_off*1e3:.2f} mm")
    print(f"  Omega: {settings.omega:.2e} rad/s")
    
    print(f"\nProfile:")
    print(f"  Sigma: [{sigma_layers[0]:.2e}, {sigma_layers[-1]:.2e}] S/m")
    print(f"  Mu: [{mu_layers[0]:.2f}, {mu_layers[-1]:.2f}]")
    
    edc = edc_forward(sigma_layers, mu_layers, settings)
    
    print(f"\nEDC Response:")
    print(f"  {edc}")
    print(f"  Real: {edc.impedance_real:.6e} Ω")
    print(f"  Imag: {edc.impedance_imag:.6e} Ω")
    print(f"  Amplitude: {edc.amplitude:.6e} Ω")
    print(f"  Phase: {edc.phase_deg:.2f}°")
    
    assert edc.frequency == settings.frequency
    assert isinstance(edc.impedance_real, float)
    assert isinstance(edc.impedance_imag, float)
    assert edc.amplitude > 0
    print("✓ EDC response structure correct")
    
    vector = edc.to_vector()
    assert len(vector) == 2
    assert vector[0] == edc.impedance_real
    assert vector[1] == edc.impedance_imag
    print("✓ Vector conversion works")
    
    edc2 = EDCResponse.from_complex(settings.frequency, complex(1.5, 2.3))
    assert edc2.impedance_real == 1.5
    assert edc2.impedance_imag == 2.3
    print("✓ Complex impedance conversion works")
    
    print("\n" + "="*70)
    print("Forward EDC Solver: PASSED ✓")
    print("="*70 + "\n")


def test_integration():
    """Test integration of all Phase 1 components."""
    print("="*70)
    print("Testing Phase 1 Integration")
    print("="*70)
    
    config = CONFIG
    
    K = config.K
    sigma = np.linspace(config.sigma_bounds[0], config.sigma_bounds[1], K)
    mu = np.linspace(config.mu_bounds[0], config.mu_bounds[1], K)
    
    normalizer = ProfileNormalizer(
        sigma_min=config.sigma_bounds[0],
        sigma_max=config.sigma_bounds[1],
        mu_min=config.mu_bounds[0],
        mu_max=config.mu_bounds[1],
        target_range=config.normalize_to_range
    )
    
    sigma_norm, mu_norm = normalizer.normalize(sigma, mu)
    print(f"\n✓ Normalized profiles using config bounds")
    
    sigma_phys, mu_phys = normalizer.denormalize(sigma_norm, mu_norm)
    
    settings = ProbeSettings(
        frequency=config.default_frequency,
        inner_radius=config.default_inner_radius,
        outer_radius=config.default_outer_radius,
        lift_off=config.default_lift_off,
        coil_height=config.default_coil_height,
        n_turns=config.default_n_turns,
    )
    
    edc = edc_forward(sigma_phys, mu_phys, settings, config.layer_thickness)
    
    print(f"✓ Calculated EDC from denormalized profiles")
    print(f"  EDC: {edc.impedance_real:.4f}{edc.impedance_imag:+.4f}j Ω")
    
    print("\n" + "="*70)
    print("Phase 1 Integration: PASSED ✓")
    print("="*70 + "\n")


def main():
    """Run all Phase 1 tests."""
    print("\n" + "="*70)
    print("PHASE 1: FOUNDATION COMPONENTS - TEST SUITE")
    print("="*70 + "\n")
    
    try:
        test_global_config()
        test_profile_normalizer()
        test_forward_edc()
        test_integration()
        
        print("\n" + "="*70)
        print("ALL PHASE 1 TESTS PASSED ✓✓✓")
        print("="*70)
        print("\nPhase 1 foundation is ready!")
        print("Next steps:")
        print("  1. Implement inverse problem solver (Phase 3)")
        print("  2. Build profile database")
        print("  3. End-to-end pipeline")
        print("\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
