import React from 'react';

interface PredictionResult {
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface ResultsPanelProps {
  result: PredictionResult;
  onFullAnalysis: () => void;
  onChatOpen: () => void;
}

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return 'from-green-400 to-emerald-500';
  if (confidence >= 0.6) return 'from-yellow-400 to-orange-500';
  return 'from-red-400 to-pink-500';
};

const getConfidenceBadgeColor = (confidence: number): string => {
  if (confidence >= 0.8) return 'bg-green-500/20 border-green-500/40 text-green-200';
  if (confidence >= 0.6) return 'bg-yellow-500/20 border-yellow-500/40 text-yellow-200';
  return 'bg-red-500/20 border-red-500/40 text-red-200';
};

export function ResultsPanel({
  result,
  onFullAnalysis,
  onChatOpen,
}: ResultsPanelProps) {
  const topProbabilities = Object.entries(result.probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Main Prediction Card */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl p-px opacity-75 group-hover:opacity-100 transition-opacity" />

        <div className="relative rounded-2xl bg-slate-900/80 backdrop-blur-xl border border-white/10 p-8 space-y-6">
          {/* Class Name - Large & Prominent */}
          <div>
            <p className="text-sm font-medium text-white/50 mb-2">Predicted Tumor Type</p>
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl blur opacity-50" />
              <div className="relative px-6 py-3 rounded-xl bg-slate-800 border border-white/10">
                <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-cyan-300">
                  {result.class_name}
                </p>
              </div>
            </div>
          </div>

          {/* Confidence Visualization */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm font-medium text-white/70">Confidence Score</p>
              <div
                className={`px-3 py-1 rounded-full border text-xs font-semibold ${getConfidenceBadgeColor(
                  result.confidence
                )}`}
              >
                {confidencePercent}%
              </div>
            </div>

            {/* Animated Progress Bar */}
            <div className="w-full bg-white/5 rounded-full h-3 overflow-hidden border border-white/10 backdrop-blur">
              <div
                className={`h-full bg-gradient-to-r ${getConfidenceColor(
                  result.confidence
                )} shadow-lg transition-all duration-700 ease-out`}
                style={{
                  width: `${confidencePercent}%`,
                  boxShadow: `0 0 20px rgba(59, 130, 246, 0.5)`,
                }}
              />
            </div>

            {/* Interpretation */}
            <p className="text-xs text-white/50 mt-2">
              {result.confidence >= 0.8
                ? '✓ High confidence prediction'
                : result.confidence >= 0.6
                ? '⚠ Moderate confidence - review recommended'
                : '❕ Low confidence - manual review suggested'}
            </p>
          </div>

          {/* Top Probabilities Bar Chart */}
          <div>
            <p className="text-sm font-medium text-white/70 mb-3">All Predictions</p>
            <div className="space-y-2">
              {topProbabilities.map(([className, probability]) => (
                <div key={className}>
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-xs font-medium text-white/70 capitalize truncate">
                      {className}
                    </p>
                    <p className="text-xs font-semibold text-white/90 ml-2">
                      {(probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden border border-white/10">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-500"
                      style={{ width: `${probability * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Full Analysis Button */}
        <button
          onClick={onFullAnalysis}
          className="group relative overflow-hidden rounded-xl bg-white/5 border border-white/10 hover:border-blue-500/50 p-4 transition-all duration-300 hover:bg-white/10"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/0 to-cyan-600/0 group-hover:from-blue-600/10 group-hover:to-cyan-600/10 transition-all duration-300" />
          <div className="relative space-y-2">
            <p className="font-semibold text-white flex items-center gap-2">
              <span>📊</span> Full Analysis
            </p>
            <p className="text-xs text-white/50">
              Deep dive with segmentation & detailed metrics
            </p>
          </div>
        </button>

        {/* Chat with AI Button */}
        <button
          onClick={onChatOpen}
          className="group relative overflow-hidden rounded-xl bg-gradient-to-r from-blue-600/20 to-cyan-600/20 border border-blue-500/50 hover:border-blue-400 p-4 transition-all duration-300 hover:from-blue-600/30 hover:to-cyan-600/30"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-cyan-600/10 group-hover:from-blue-600/20 group-hover:to-cyan-600/20 transition-all duration-300" />
          <div className="relative space-y-2">
            <p className="font-semibold text-blue-200 flex items-center gap-2">
              <span>💬</span> Chat with AI
            </p>
            <p className="text-xs text-blue-300/70">
              Ask questions about this diagnosis
            </p>
          </div>
        </button>
      </div>

      {/* Medical Disclaimer */}
      <div className="rounded-xl bg-amber-500/10 border border-amber-500/30 p-4 backdrop-blur">
        <p className="text-xs text-amber-200/80 leading-relaxed">
          <span className="font-semibold">⚕️ Medical Disclaimer:</span> This AI
          tool provides diagnostic assistance for research and educational
          purposes. It is NOT a substitute for professional medical diagnosis.
          Always consult qualified healthcare professionals for clinical
          decisions.
        </p>
      </div>
    </div>
  );
}
