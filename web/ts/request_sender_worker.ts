import {RequestSender} from "./request_sender_background";

// 声明 onconnect 是 SharedWorker 的全局事件
declare let onconnect: (e: MessageEvent) => void;
// onconnect 事件处理函数
onconnect = (e:any) => {
    const port = e.ports[0];
    port.start();

    const on_response_ok=(response_data: any) =>{
        port.postMessage({ action: 'response_ok', response_data });
}

    const on_response_error=(error_status: number|string) =>{
        port.postMessage({ action: 'response_error', error: 'HTTP error! status: ' + error_status });
    }
    // 监听来自页面的消息
    port.onmessage = (event:any) => {
        const { action, request, options, token } = event.data || {};
        const sender = new RequestSender(token);

        if (!action) {
            console.error('Invalid message: missing action');
            return;
        }

        if (action === 'makeRequest') {
            if (!request || typeof request !== 'object') {
                console.error('Invalid request format');
                return;
            }

            // 确保 options 存在并包含必要的字段
            if (!options || typeof options !== 'object') {
                console.error('Invalid options format');
                return;
            }

            // 处理请求并捕获错误
            try {
                sender.makeRequest(
                        request.method,
                        request.url,
                        request.params,
                        request.body,
                        token ?request.useToken : false,// 如果有 token 则使用，否则关闭 useToken
                        on_response_ok,
                        on_response_error,
                );
            } catch (error) {
                console.error('Failed to make request:', error);
                port.postMessage({ action: 'response_error', error: (error as Error).message });
            }
        } else {
            console.warn(`Unrecognized action: ${action}`);
        }
    };

    // 通知页面 SharedWorker 已连接
    port.postMessage({ action: 'request_sender_worker_started' });
};