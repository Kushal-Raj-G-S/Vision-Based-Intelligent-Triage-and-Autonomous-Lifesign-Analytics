// VITAL has a single theme (warm white + orange, defined in app/globals.css). This init script
// only pins <html data-theme="light"> before first paint and clears any old saved dark choice.
export const THEME_INIT_SCRIPT = `(function(){try{localStorage.removeItem('vital-theme');}catch(e){}document.documentElement.dataset.theme='light';})();`;
