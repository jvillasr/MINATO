import unittest
import numpy as np
from scipy.integrate import quad
from minato.binary_population.joint_outcomes import (
    OutcomeAxis, OutcomeBank, JointSurveyOutcomeLikelihood,
    powerlaw_logpdf, mass_ratio_efficiency, integrate_mass_ratio_response,
)


class TestJointOutcomes(unittest.TestCase):
    def test_powerlaw_normalisation_including_minus_one(self):
        for exponent in [-3.,-1.,-1+1e-12,0.,2.]:
            integral=quad(lambda x:np.exp(powerlaw_logpdf(x,exponent,.1,3.5)),.1,3.5)[0]
            self.assertAlmostEqual(integral,1.,places=10)

    def make(self, axes):
        values={'rv':np.array([0.,0.,1.,1.]),'period':np.array([np.nan,np.nan,2.,2.])}
        single=OutcomeBank({'rv':np.array([0.,0.]),'period':np.array([np.nan,np.nan])},np.zeros(2,int))
        binary=OutcomeBank(values,np.zeros(4,int),np.ones(4),np.array([.2,.4,.6,.8]),np.zeros(4))
        return JointSurveyOutcomeLikelihood(values,np.zeros(4,int),single,binary,axes,(.1,3.),(.1,1.))

    def test_correlated_period_does_not_double_count(self):
        rv=OutcomeAxis('rv',(-.5,.5,1.5))
        period=OutcomeAxis('period',(1.,3.),missing=True)
        base=self.make([rv]);joint=self.make([rv,period])
        # Perfectly redundant periods cannot add independent likelihood terms.
        for f in [.2,.7,1.]:
            self.assertAlmostEqual(base(f,.3),joint(f,.3))
        self.assertEqual(joint.diagnostics(.7,0)['observed_systems'],4)

    def test_missing_outcomes_and_marginalisation(self):
        rv=OutcomeAxis('rv',(-.5,.5,1.5));period=OutcomeAxis('period',(1.,3.),True)
        base=self.make([rv]);joint=self.make([rv,period])
        p,_=joint.probabilities(.7,.2,.5);b,_=base.probabilities(.7,.2,.5)
        np.testing.assert_allclose(p.reshape(1,2,2).sum(axis=2),b)
        self.assertTrue(np.isfinite(joint(.7,0)))

    def test_importance_reweighting_changes_q_sensitive_outcomes(self):
        likelihood=self.make([OutcomeAxis('rv',(-.5,.5,1.5))])
        a,_=likelihood.probabilities(1,0,-1);b,_=likelihood.probabilities(1,0,1)
        self.assertGreater(b[0,1],a[0,1])
        np.testing.assert_allclose(a.sum(),1)

    def test_unsupported_cells_are_not_silently_floored(self):
        likelihood=self.make([OutcomeAxis('rv',(-.5,.5,1.5))])
        self.assertEqual(likelihood(0,0),-np.inf)
        self.assertEqual(likelihood.diagnostics(0,0)['observed_unsupported_cells'],1)

    def test_conditions_require_support(self):
        with self.assertRaises(ValueError):
            JointSurveyOutcomeLikelihood({'rv':[0.]},np.array([1]),
                OutcomeBank({'rv':[0.]},np.array([0])),
                OutcomeBank({'rv':[0.]},np.array([0]),[1.],[.5],[0.]),
                [OutcomeAxis('rv',(-1.,1.))],(.1,3.),(.1,1.))

    def test_missing_and_out_of_bounds_are_distinct(self):
        a=OutcomeAxis('q',(-np.inf,.3,.6,np.inf),True)
        np.testing.assert_array_equal(a.encode([np.nan,.2,.5,1.1]),[0,1,2,3])
        with self.assertRaises(ValueError):OutcomeAxis('q',(.1,1.),True).encode([1.1])

    def test_q_efficiency_scenarios(self):
        for maximum in [.25,.5,1.]:
            np.testing.assert_allclose(mass_ratio_efficiency([.1,.3,.65,1.],maximum),[0,0,maximum/2,maximum])


class TestIntegratedMassRatioResponse(unittest.TestCase):
    def test_marginal_probability_and_ess_are_preserved(self):
        binary=OutcomeBank({'rv':np.array([0.,1.])},np.zeros(2,int),
                           np.array([1.,2.]),np.array([.4,.9]),np.zeros(2))
        axis=OutcomeAxis('q',(-np.inf,.4,.7,np.inf),True)
        split=integrate_mass_ratio_response(binary,axis,.5)
        self.assertAlmostEqual(split.response_weight.sum(),2.)
        single=OutcomeBank({'rv':np.array([0.]),'q':np.array([np.nan])},np.zeros(1,int))
        rv=OutcomeAxis('rv',(-.5,.5,1.5))
        obs={'rv':np.array([0.,1.]),'q':np.array([np.nan,.8])}
        model=JointSurveyOutcomeLikelihood(obs,np.zeros(2,int),single,split,[rv,axis],(.1,3.),(.1,1.))
        original=JointSurveyOutcomeLikelihood(obs,np.zeros(2,int),single,binary,[rv],(.1,3.),(.1,1.))
        p,e=model.probabilities(.7,.3,-.2);p0,e0=original.probabilities(.7,.3,-.2)
        np.testing.assert_allclose(p.reshape(1,2,4).sum(axis=2),p0)
        np.testing.assert_allclose(e,e0)

    def test_lower_cutoff_and_finer_bins_preserve_selection(self):
        bank=OutcomeBank({'rv':[0.,1.,2.]},np.zeros(3,int),[1.,2.,2.5],[.1,.2,.8],[0.,0.,0.])
        axis=OutcomeAxis('q',tuple(np.r_[-np.inf,np.arange(.1,1.001,.05),np.inf]),True)
        split=integrate_mass_ratio_response(bank,axis,.5,minimum_q=.1)
        measured=np.isfinite(split.values['q'])
        recovered=np.bincount(split.source_index[measured],weights=split.response_weight[measured],minlength=3)
        np.testing.assert_allclose(recovered,[0.,.5/9,.5*7/9])
        np.testing.assert_allclose(np.bincount(split.source_index,weights=split.response_weight),1.)
        old=integrate_mass_ratio_response(bank,axis,.5)
        explicit=integrate_mass_ratio_response(bank,axis,.5,minimum_q=.3)
        np.testing.assert_array_equal(old.response_weight,explicit.response_weight)

    def test_ineligible_or_zero_efficiency_is_always_missing(self):
        bank=OutcomeBank({'rv':[0.,1.]},np.zeros(2,int),[1.,2.],[.5,.9],[0.,0.])
        axis=OutcomeAxis('q',(-np.inf,.5,np.inf),True)
        for maximum,eligible in [(0.,[True,True]),(1.,[False,False])]:
            split=integrate_mass_ratio_response(bank,axis,maximum,eligible=eligible)
            self.assertTrue(np.isnan(split.values['q']).all())
            np.testing.assert_allclose(split.response_weight,[1.,1.])

if __name__=='__main__':unittest.main()
