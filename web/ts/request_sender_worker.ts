import {RequestMessage, RequestSender} from "./request_sender_background";

// 声明 onconnect 是 SharedWorker 的全局事件
declare let onconnect: (e: MessageEvent) => void;
// onconnect 事件处理函数
onconnect = (e:any) => {
    const port = e.ports[0];
    port.start();

    const on_response_ok=(response_data: any) =>{
        port.postMessage({ action: 'on_response_ok', response_data:response_data });
    }

    const on_response_error=(error: string) =>{
        port.postMessage({ action: 'on_response_error', error: error });
    }
    // 监听来自页面的消息
    port.onmessage = (event:any) => {
        const { action, request, token } = event.data || {};
        const sender = new RequestSender();

        if (!action) {
            console.error('Invalid message: missing action');
            return;
        }

        if (action === 'makeRequest') {
            if (!request || typeof request !== 'object') {
                console.error('Invalid request format');
                return;
            }
            const params = new URLSearchParams(request.paramsObject);
            const request_message:RequestMessage={
                method:request.method,
                url:request.url,
                params:params,
                body:request.body,
                token:request.token,
                content_type:request.content_type,
                on_response_ok:(response_data: any)=>on_response_ok(response_data),
                on_response_error:(error: string)=>on_response_error(error)
            }
            // 处理请求并捕获错误
            try {
                sender.makeRequest(request_message);
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