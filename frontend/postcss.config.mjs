// Tailwind v4 plugs into PostCSS through this package rather than a
// tailwind.config.js — configuration lives in CSS now (see app/globals.css).
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
