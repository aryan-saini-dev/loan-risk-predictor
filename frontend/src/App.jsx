import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Calculator, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    age: '',
    income: '',
    loan_amount: '',
    credit_score: '',
    employment_years: '',
    education_level: '',
    housing_status: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${apiUrl}/predict`, {
        age: parseInt(formData.age),
        income: parseFloat(formData.income),
        loan_amount: parseFloat(formData.loan_amount),
        credit_score: parseInt(formData.credit_score),
        employment_years: parseFloat(formData.employment_years),
        education_level: parseInt(formData.education_level),
        housing_status: parseInt(formData.housing_status)
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      age: '',
      income: '',
      loan_amount: '',
      credit_score: '',
      employment_years: '',
      education_level: '',
      housing_status: ''
    });
    setResult(null);
    setError('');
  };

  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden">
      {/* Noise Texture */}
      <svg className="noise-texture" xmlns="http://www.w3.org/2000/svg">
        <filter id="noise">
          <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="4" />
        </filter>
        <rect width="100%" height="100%" filter="url(#noise)" />
      </svg>

      {/* Hero Section with Marquee */}
      <div className="relative z-10">
        <div className="bg-accent py-8 overflow-hidden">
          <div className="flex marquee whitespace-nowrap">
            <div className="flex items-center space-x-8 px-4">
              <span className="text-[6rem] md:text-[8rem] font-bold text-accent-foreground">RISK</span>
              <span className="text-2xl md:text-4xl font-bold text-accent-foreground">ANALYSIS</span>
              <span className="text-[6rem] md:text-[8rem] font-bold text-accent-foreground">PREDICT</span>
              <span className="text-2xl md:text-4xl font-bold text-accent-foreground">APPROVE</span>
              <span className="text-[6rem] md:text-[8rem] font-bold text-accent-foreground">DENY</span>
              <span className="text-2xl md:text-4xl font-bold text-accent-foreground">SCORE</span>
            </div>
            <div className="flex items-center space-x-8 px-4">
              <span className="text-[6rem] md:text-[8rem] font-bold text-accent-foreground">RISK</span>
              <span className="text-2xl md:text-4xl font-bold text-accent-foreground">ANALYSIS</span>
              <span className="text-[6rem] md:text-[8rem] font-bold text-accent-foreground">PREDICT</span>
              <span className="text-2xl md:text-4xl font-bold text-accent-foreground">APPROVE</span>
              <span className="text-[6rem] md:text-[8rem] font-bold text-accent-foreground">DENY</span>
              <span className="text-2xl md:text-4xl font-bold text-accent-foreground">SCORE</span>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-20 md:py-32">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h1 className="text-[clamp(3rem,12vw,6rem)] font-bold uppercase tracking-tighter leading-none mb-6">
              LOAN DEFAULT
              <span className="text-accent"> RISK</span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto">
              AI-POWERED PREDICTION SYSTEM FOR FINANCIAL RISK ASSESSMENT
            </p>
          </motion.div>

          {/* Form Section */}
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="max-w-4xl mx-auto"
          >
            <div className="border-2 border-border p-8 md:p-12">
              <form onSubmit={handleSubmit} className="space-y-8">
                <div className="grid md:grid-cols-2 gap-8">
                  {/* Age */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      AGE
                    </label>
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground placeholder-muted focus:border-accent outline-none transition-colors"
                      placeholder="25"
                      required
                      min="18"
                      max="100"
                    />
                  </div>

                  {/* Income */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      ANNUAL INCOME
                    </label>
                    <input
                      type="number"
                      name="income"
                      value={formData.income}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground placeholder-muted focus:border-accent outline-none transition-colors"
                      placeholder="50000"
                      required
                      min="0"
                      step="0.01"
                    />
                  </div>

                  {/* Loan Amount */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      LOAN AMOUNT
                    </label>
                    <input
                      type="number"
                      name="loan_amount"
                      value={formData.loan_amount}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground placeholder-muted focus:border-accent outline-none transition-colors"
                      placeholder="10000"
                      required
                      min="0"
                      step="0.01"
                    />
                  </div>

                  {/* Credit Score */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      CREDIT SCORE
                    </label>
                    <input
                      type="number"
                      name="credit_score"
                      value={formData.credit_score}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground placeholder-muted focus:border-accent outline-none transition-colors"
                      placeholder="650"
                      required
                      min="300"
                      max="850"
                    />
                  </div>

                  {/* Employment Years */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      EMPLOYMENT YEARS
                    </label>
                    <input
                      type="number"
                      name="employment_years"
                      value={formData.employment_years}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground placeholder-muted focus:border-accent outline-none transition-colors"
                      placeholder="5"
                      required
                      min="0"
                      step="0.1"
                    />
                  </div>

                  {/* Education Level */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      EDUCATION LEVEL
                    </label>
                    <select
                      name="education_level"
                      value={formData.education_level}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground focus:border-accent outline-none transition-colors"
                      required
                    >
                      <option value="" className="bg-background">SELECT</option>
                      <option value="0" className="bg-background">HIGH SCHOOL</option>
                      <option value="1" className="bg-background">BACHELORS</option>
                      <option value="2" className="bg-background">MASTERS</option>
                      <option value="3" className="bg-background">PHD</option>
                    </select>
                  </div>

                  {/* Housing Status */}
                  <div>
                    <label className="block text-sm uppercase tracking-widest text-muted-foreground mb-2">
                      HOUSING STATUS
                    </label>
                    <select
                      name="housing_status"
                      value={formData.housing_status}
                      onChange={handleInputChange}
                      className="w-full h-24 bg-transparent border-b-2 border-border text-4xl font-bold uppercase tracking-tight text-foreground focus:border-accent outline-none transition-colors"
                      required
                    >
                      <option value="" className="bg-background">SELECT</option>
                      <option value="0" className="bg-background">RENT</option>
                      <option value="1" className="bg-background">MORTGAGE</option>
                      <option value="2" className="bg-background">OWN</option>
                    </select>
                  </div>
                </div>

                {/* Submit Button */}
                <div className="flex justify-center pt-8">
                  <button
                    type="submit"
                    disabled={loading}
                    className="group relative bg-accent text-accent-foreground px-12 py-6 text-xl font-bold uppercase tracking-tighter border-0 hover:scale-105 active:scale-95 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'ANALYZING...' : 'PREDICT RISK'}
                    <div className="absolute inset-0 bg-foreground scale-0 group-hover:scale-100 transition-transform duration-300 -z-10"></div>
                    <span className="relative z-10">{loading ? 'ANALYZING...' : 'PREDICT RISK'}</span>
                  </button>
                </div>
              </form>

              {/* Error Display */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-8 p-6 border-2 border-red-500 bg-red-500/10"
                >
                  <div className="flex items-center space-x-3">
                    <AlertCircle className="w-6 h-6 text-red-500" />
                    <p className="text-red-500 font-bold uppercase">{error}</p>
                  </div>
                </motion.div>
              )}

              {/* Result Display */}
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-8"
                >
                  <div className={`p-8 border-2 ${result.prediction === 0 ? 'border-green-500 bg-green-500/10' : 'border-red-500 bg-red-500/10'}`}>
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-4">
                        {result.prediction === 0 ? (
                          <>
                            <CheckCircle className="w-8 h-8 text-green-500" />
                            <div>
                              <h3 className="text-3xl font-bold uppercase text-green-500">{result.risk_level}</h3>
                              <p className="text-xl text-green-400">{result.status}</p>
                            </div>
                          </>
                        ) : (
                          <>
                            <AlertCircle className="w-8 h-8 text-red-500" />
                            <div>
                              <h3 className="text-3xl font-bold uppercase text-red-500">{result.risk_level}</h3>
                              <p className="text-xl text-red-400">{result.status}</p>
                            </div>
                          </>
                        )}
                      </div>
                      <div className="text-right">
                        <div className="text-5xl font-bold">
                          {result.prediction === 0 ? (
                            <span className="text-green-500">{(result.probability * 100).toFixed(1)}%</span>
                          ) : (
                            <span className="text-red-500">{(result.probability * 100).toFixed(1)}%</span>
                          )}
                        </div>
                        <p className="text-sm uppercase tracking-widest text-muted-foreground">RISK PROBABILITY</p>
                      </div>
                    </div>

                    <div className="flex justify-center pt-6">
                      <button
                        onClick={resetForm}
                        className="bg-border text-foreground px-8 py-4 text-lg font-bold uppercase tracking-tighter border-2 border-border hover:bg-foreground hover:text-background transition-all duration-300"
                      >
                        NEW ASSESSMENT
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

export default App;
