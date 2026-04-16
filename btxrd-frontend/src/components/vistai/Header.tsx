export function Header() {
  return (
    <header className="relative z-20 border-b border-white/10 bg-gradient-to-r from-slate-900/80 via-blue-900/40 to-slate-900/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between">
          {/* Logo & Branding */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <span className="text-white font-bold text-lg">V</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">
                VistAI
              </h1>
              <p className="text-xs text-blue-300/80">
                Bone Tumor Detection & Analysis
              </p>
            </div>
          </div>

          {/* Status badge */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-green-500/30 backdrop-blur">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
            <span className="text-xs font-medium text-green-300">
              AI Ready
            </span>
          </div>
        </div>

        {/* Description */}
        <p className="mt-3 text-sm text-white/60 max-w-2xl">
          Advanced medical AI for bone tumor classification, segmentation, and diagnosis using knowledge-distilled deep learning models.
        </p>
      </div>
    </header>
  );
}
