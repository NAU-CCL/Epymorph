document.addEventListener("DOMContentLoaded", function () {
    // Regex to match Python-style fully-qualified module names:
    // e.g., `foo.bar.baz_qux.quux`
    const moduleName = /^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$/;
    
    // The module name headers can get quite long and we want them to break legibly.
    // Insert word break opporunities (<wbr/>) before each dot in the module name.
    document
        .querySelectorAll("div.md-content > article > h1:first-child")
        .forEach(h1 => {
            const text = h1.textContent.trim();
            if (moduleName.test(text)) {
                h1.innerHTML = text.split(".").join("<wbr>.");
            }
        });
});
