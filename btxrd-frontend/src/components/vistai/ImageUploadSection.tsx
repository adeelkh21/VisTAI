import React, { useRef } from 'react';

interface ImageUploadSectionProps {
  preview: string | null;
  loading: boolean;
  onFileSelect: (file: File) => void;
  onRunAnalysis: () => void;
}

export function ImageUploadSection({
  preview,
  loading,
  onFileSelect,
  onRunAnalysis,
}: ImageUploadSectionProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-blue-500', 'bg-blue-500/5');
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.currentTarget.classList.remove('border-blue-500', 'bg-blue-500/5');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-blue-500', 'bg-blue-500/5');
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className="relative group cursor-pointer"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl p-px">
          <div className="absolute inset-0 bg-slate-900 rounded-2xl opacity-90 group-hover:opacity-80 transition-opacity" />
        </div>

        <div className="relative px-8 py-12 rounded-2xl border border-white/10 backdrop-blur-xl transition-all duration-300 hover:border-blue-500/50">
          {preview ? (
            <div className="flex flex-col items-center gap-4">
              <div className="relative w-full">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-lg"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="absolute top-2 right-2 p-2 rounded-lg bg-slate-900/80 hover:bg-slate-800 transition-colors"
                >
                  <svg
                    className="w-5 h-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                  </svg>
                </button>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
                className="text-xs text-blue-300 hover:text-blue-200 transition-colors"
              >
                Click to change image
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="text-5xl">📸</div>
              <div>
                <p className="text-white font-medium">
                  Click to upload or drag image
                </p>
                <p className="text-sm text-white/50 mt-1">
                  PNG, JPG, JPEG up to 16MB
                </p>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            aria-label="Upload X-ray image"
          />
        </div>
      </div>

      {/* Run Analysis Button */}
      <button
        onClick={onRunAnalysis}
        disabled={loading || !preview}
        className="w-full relative group"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl blur opacity-75 group-hover:opacity-100 transition duration-300 disabled:opacity-50" />
        <div className="relative px-6 py-4 bg-slate-900 rounded-xl transition-all duration-200 group-hover:scale-105 disabled:scale-100">
          <span className="text-white font-semibold flex items-center justify-center gap-2">
            {loading ? (
              <>
                <svg
                  className="animate-spin w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Analyzing...
              </>
            ) : (
              <>
                <span>🔍</span>
                Run Analysis
              </>
            )}
          </span>
        </div>
      </button>

      {/* Info cards */}
      <div className="space-y-3">
        <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 backdrop-blur">
          <p className="text-sm text-blue-100">
            <span className="font-semibold">💡 Tip:</span> Upload high-quality X-ray images for best results
          </p>
        </div>
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur">
          <p className="text-xs text-white/60">
            Supports 9 bone tumor types: Giant cell, Osteochondroma, Osteosarcoma, and more.
          </p>
        </div>
      </div>
    </div>
  );
}
