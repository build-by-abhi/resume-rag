/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Small palette so the UI reads as one system instead of ad-hoc colours.
        brand: {
          50: "#eef4ff",
          100: "#d9e5ff",
          200: "#bcd2ff",
          300: "#8eb5ff",
          400: "#598eff",
          500: "#3366ff",
          600: "#1f47f5",
          700: "#1834e1",
          800: "#1a2eb6",
          900: "#1c2e8f",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};