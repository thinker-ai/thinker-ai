// 动态加载 CSS 文件
const link = document.createElement('link');
link.rel = 'stylesheet';
link.href = chrome.runtime.getURL('css/floating-panel.css');
document.head.appendChild(link);

// 动态加载 floating-panel.js
const script = document.createElement('script');
script.src = chrome.runtime.getURL('js/floating-panel.js');
document.body.appendChild(script);


function highlightIframeContent() {
  const iframes = document.querySelectorAll('iframe');
  iframes.forEach((iframe) => {
    const rect = iframe.getBoundingClientRect();
    const highlightDiv = document.createElement('div');
    highlightDiv.classList.add('highlight-iframe');
    highlightDiv.style.left = `${rect.left}px`;
    highlightDiv.style.top = `${rect.top}px`;
    highlightDiv.style.width = `${rect.width}px`;
    highlightDiv.style.height = `${rect.height}px`;
    document.body.appendChild(highlightDiv);
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "highlightIframes") {
    highlightIframeContent();
  }
});