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

        switch (action) {
            case 'connect':
                console.log("web_socket_sender_worker action is connect");
                web_socket_sender_worker.connect(content);   // 调用包含 token 的 connect 函数
                break;

            case 'send_message':
                console.log("web_socket_sender_worker action is send_message");
                if (web_socket_sender_worker.socket && web_socket_sender_worker.socket.readyState === WebSocket.OPEN) {
                    web_socket_sender_worker.send_message(content);
                } else {
                    web_socket_sender_worker.on_send_error('Socket is not open');
                }
                break;

            case 'register_function_listener':
                const { matchingFunction, callbackId } = content;
                // 注册带有匹配函数的监听器
                web_socket_sender_worker.register_function_listener(matchingFunction, callbackId);
                break;

            case 'register_key_listener':
                const { key, callbackId: callbackIdKey } = content;  // 修改 callbackId 的别名以避免冲突
                // 通过 key 注册监听器
                web_socket_sender_worker.register_key_listener(key, callbackIdKey);
                break;

            default:
                console.warn(`Unrecognized action: ${action}`);
                break;
        }
    };
    // 通知页面 SharedWorker 已连接
    port.postMessage({ action: 'web_socket_worker_started' });
};