"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { User, X } from "lucide-react";

interface PatientInfoDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (info: PatientInfo) => void;
}

export interface PatientInfo {
  caseId: string;
  patientName: string;
  patientAge: string;
  clinicalIndication: string;
}

export default function PatientInfoDialog({ isOpen, onClose, onSubmit }: PatientInfoDialogProps) {
  const [caseId, setCaseId] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [clinicalIndication, setClinicalIndication] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!caseId.trim()) {
      newErrors.caseId = "Case ID is required";
    }
    
    if (!patientName.trim()) {
      newErrors.patientName = "Patient name is required";
    }
    
    if (!patientAge.trim()) {
      newErrors.patientAge = "Patient age is required";
    } else {
      const age = parseInt(patientAge);
      if (isNaN(age) || age < 0 || age > 150) {
        newErrors.patientAge = "Please enter a valid age (0-150)";
      }
    }
    
    if (!clinicalIndication.trim()) {
      newErrors.clinicalIndication = "Clinical indication is required";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    onSubmit({
      caseId: caseId.trim(),
      patientName: patientName.trim(),
      patientAge: patientAge.trim(),
      clinicalIndication: clinicalIndication.trim(),
    });
    
    // Reset form
    setCaseId("");
    setPatientName("");
    setPatientAge("");
    setClinicalIndication("");
    setErrors({});
  };

  const handleClose = () => {
    setCaseId("");
    setPatientName("");
    setPatientAge("");
    setClinicalIndication("");
    setErrors({});
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
          />

          {/* Dialog */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="bg-[#0f172a] border border-[#1f2937] rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden"
            >
              {/* Header */}
              <div className="px-6 py-4 border-b border-[#1f2937] bg-[#111827] flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                    <User className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-[#f8fafc]">Patient Information</h3>
                    <p className="text-xs text-[#64748b]">Required for professional radiology report</p>
                  </div>
                </div>
                <button
                  onClick={handleClose}
                  className="text-[#64748b] hover:text-[#e2e8f0] transition-colors p-1 rounded-lg hover:bg-[#1e293b]"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Form */}
              <form onSubmit={handleSubmit} className="p-6 space-y-5">
                {/* Case ID */}
                <div>
                  <label htmlFor="caseId" className="block text-sm font-medium text-[#e2e8f0] mb-2">
                    Case ID <span className="text-red-400">*</span>
                  </label>
                  <input
                    id="caseId"
                    type="text"
                    value={caseId}
                    onChange={(e) => setCaseId(e.target.value)}
                    placeholder="Enter case/accession number"
                    className={`w-full px-4 py-2.5 bg-[#1e293b] border rounded-lg text-[#f8fafc] placeholder:text-[#475569]
                             focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all
                             ${errors.caseId ? "border-red-500" : "border-[#334155]"}`}
                  />
                  {errors.caseId && (
                    <p className="mt-1.5 text-xs text-red-400">{errors.caseId}</p>
                  )}
                </div>

                {/* Patient Name */}
                <div>
                  <label htmlFor="patientName" className="block text-sm font-medium text-[#e2e8f0] mb-2">
                    Patient Name <span className="text-red-400">*</span>
                  </label>
                  <input
                    id="patientName"
                    type="text"
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    placeholder="Enter full name"
                    className={`w-full px-4 py-2.5 bg-[#1e293b] border rounded-lg text-[#f8fafc] placeholder:text-[#475569]
                             focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all
                             ${errors.patientName ? "border-red-500" : "border-[#334155]"}`}
                  />
                  {errors.patientName && (
                    <p className="mt-1.5 text-xs text-red-400">{errors.patientName}</p>
                  )}
                </div>

                {/* Patient Age */}
                <div>
                  <label htmlFor="patientAge" className="block text-sm font-medium text-[#e2e8f0] mb-2">
                    Patient Age <span className="text-red-400">*</span>
                  </label>
                  <input
                    id="patientAge"
                    type="number"
                    value={patientAge}
                    onChange={(e) => setPatientAge(e.target.value)}
                    placeholder="Enter age in years"
                    min="0"
                    max="150"
                    className={`w-full px-4 py-2.5 bg-[#1e293b] border rounded-lg text-[#f8fafc] placeholder:text-[#475569]
                             focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all
                             ${errors.patientAge ? "border-red-500" : "border-[#334155]"}`}
                  />
                  {errors.patientAge && (
                    <p className="mt-1.5 text-xs text-red-400">{errors.patientAge}</p>
                  )}
                </div>

                {/* Clinical Indication */}
                <div>
                  <label htmlFor="clinicalIndication" className="block text-sm font-medium text-[#e2e8f0] mb-2">
                    Clinical Indication <span className="text-red-400">*</span>
                  </label>
                  <textarea
                    id="clinicalIndication"
                    value={clinicalIndication}
                    onChange={(e) => setClinicalIndication(e.target.value)}
                    placeholder="e.g., Suspected bone tumor, pain in left femur, follow-up examination..."
                    rows={4}
                    className={`w-full px-4 py-2.5 bg-[#1e293b] border rounded-lg text-[#f8fafc] placeholder:text-[#475569]
                             focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all resize-none
                             ${errors.clinicalIndication ? "border-red-500" : "border-[#334155]"}`}
                  />
                  {errors.clinicalIndication && (
                    <p className="mt-1.5 text-xs text-red-400">{errors.clinicalIndication}</p>
                  )}
                </div>

                {/* Info Notice */}
                <div className="p-3 rounded-lg bg-blue-950/30 border border-blue-800/30">
                  <p className="text-xs text-blue-200">
                    ℹ️ This information will be included in the professional radiology report and PDF download.
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <button
                    type="button"
                    onClick={handleClose}
                    className="flex-1 px-4 py-2.5 rounded-lg border border-[#334155] text-[#94a3b8] font-medium
                             hover:bg-[#1e293b] transition-all"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="flex-1 px-4 py-2.5 rounded-lg bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold
                             shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 transition-all"
                  >
                    Generate Report
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
