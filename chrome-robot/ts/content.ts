window.addEventListener('message', function(event) {
    if (event.source === window && event.data && event.data.action === 'isThinkerAIInstall') {
        window.postMessage({ action: 'thinkerAI', value: true }, '*');
    }
});
window.addEventListener('message', (event) => {
    if (event.data && event.data.action === 'getAuthorization') {
        // 向 background script 发送消息
        chrome.runtime.sendMessage({ action: 'getAuthorization' }, (response) => {
            // 将授权信息返回给普通网页
            window.postMessage({ action: 'authorizationResult', response: response }, '*');
        });
    }
});


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'ai_action') {
        // 根据指令执行相应的操作
        executeAction(message.data);
    }
});


function executeAction(data:any){
    chrome.runtime.sendMessage({ action: 'ai_action_result', data: data });
}
