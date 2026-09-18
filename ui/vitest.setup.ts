import "@testing-library/jest-dom/vitest";

// jsdom не реализует scrollIntoView — ленты субтитров его вызывают.
if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {
    /* no-op в тестах */
  };
}
