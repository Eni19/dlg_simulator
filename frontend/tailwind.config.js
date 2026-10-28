/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Compatibilidade dos componentes existentes com a paleta do caderno DLG / lab
        emerald: {
          50: "#e9f3f1",
          100: "#d7e9e6",
          200: "#b0d3ce",
          300: "#2a7a82",
          400: "#237078",
          500: "#1d6a72",
          600: "#175a61",
          700: "#134b51",
          800: "#dbe9e5",
          900: "#e6efec",
          950: "#f0f6f3",
        },
        graphite: {
          50: "#1f2a30",
          100: "#1f2a30",
          200: "#2e3a3f",
          300: "#3f4b4f",
          400: "#57625f",
          500: "#5b6562",
          600: "#646d6a",
          700: "#d4cbbb",
          800: "#e9e4da",
          900: "#f5f2ec",
          950: "#fffdf9",
        },
      },
      fontFamily: {
        sans: ["Figtree", "system-ui", "sans-serif"],
        serif: ["Newsreader", "Georgia", "serif"],
        hand: ["Caveat", "cursive"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.24s cubic-bezier(0.23, 1, 0.32, 1)",
        "slide-up": "slideUp 0.28s cubic-bezier(0.23, 1, 0.32, 1)",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
        glow: "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
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
