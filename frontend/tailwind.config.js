/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Compatibilidade dos componentes existentes com a paleta clara DLG / lab
        emerald: {
          50: "#edf5f6",
          100: "#dcebef",
          200: "#b5d4dc",
          300: "#257084",
          400: "#1b6c80",
          500: "#166477",
          600: "#15586a",
          700: "#164d5d",
          800: "#d4e2e7",
          900: "#dfe9ed",
          950: "#edf4f6",
        },
        graphite: {
          50: "#172b36",
          100: "#172b36",
          200: "#293e49",
          300: "#3b515d",
          400: "#526570",
          500: "#596b76",
          600: "#657681",
          700: "#c8d2d8",
          800: "#e2e8ec",
          900: "#f1f5f6",
          950: "#ffffff",
        },
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
        glow: "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
        glow: {
          "0%": { boxShadow: "0 0 5px rgba(16, 185, 129, 0.2)" },
          "100%": { boxShadow: "0 0 20px rgba(16, 185, 129, 0.4)" },
        },
      },
    },
  },
  plugins: [],
};
