import {WebSocketSenderBackgroundWithCallback} from "./web_socket_background";

declare let onconnect: (e: MessageEvent) => void;
onconnect = (e) => {
    const port = e.ports[0];
    port.start();
    const send_response= (response?: any) => {
        port.postMessage(response)
    };
    const web_socket_sender_worker = new WebSocketSenderBackgroundWithCallback(send_response)
    port.onmessage = (event) => {
        const { action, content } = event.data;
        if (action === 'connect') {
            web_socket_sender_worker.connect(content);   // 调用包含 token 的 connect 函数
        }
        // 处理发送消息的逻辑
        if (action === 'sendMessage') {
            if (web_socket_sender_worker.socket && web_socket_sender_worker.socket.readyState === WebSocket.OPEN) {
                web_socket_sender_worker.sendMessage(content)
            } else {
                web_socket_sender_worker.on_send_error('Socket is not open');
            }
        }
        if (action === 'registerFunctionListener') {
            const { matchingFunction, callbackId } = content;
            // 注册带有匹配函数的监听器
            web_socket_sender_worker.registerFunctionListener(matchingFunction, callbackId);
        } else if (action === 'registerKeyListener') {
            const { key, callbackId } = content;
            // 通过 key 注册监听器
            web_socket_sender_worker.registerKeyListener(key, callbackId);
        } else if (action === 'notifyListeners') {
            const { data } = content;

            // 通知所有注册的监听器
            web_socket_sender_worker.notifyListeners(data);
        }
    };
    // 通知页面 SharedWorker 已连接
    port.postMessage({ action: 'web socket worker started' });
};