window.addEventListener('message', (event) => {
    if (event.data && event.data.action === 'getAuthorization') {
        // 向 background script 发送消息
        chrome.runtime.sendMessage({ action: 'getAuthorization' }, (response) => {
            // 将授权信息返回给普通网页
            window.postMessage({ action: 'authorizationResult', response: response }, '*');
        });
    }
});