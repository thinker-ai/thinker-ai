// 导入 axios 类型
export type RequestMessage = {
        method: 'get' | 'post',
        url: string,
        params?: URLSearchParams,
        body?: any,
        token?:string,
        content_type?:string,
        on_response_ok?:(response_data: any) => void,
        on_response_error?:(error: string) => void,
};
// 定义请求配置的接口
export interface RequestSenderInterface {
    send_http(message:RequestMessage): void;
}
export class RequestSender implements RequestSenderInterface {
    public send_http(message: RequestMessage): void {
        let url = message.url;

        // 处理查询参数，将 token 添加到 paramsObject
        const paramsObject = message.params ? Object.fromEntries(message.params.entries()) : {};
        if (message.token) {
            paramsObject["token"] = message.token; // 将 token 添加到查询参数
        }

        // 构建最终 URL
        const queryString = new URLSearchParams(paramsObject).toString();
        if (queryString) {
            url += (url.includes('?') ? '&' : '?') + queryString;
        }

        const headers: Record<string, string> = {};

        if (message.content_type) {
            headers['Content-Type'] = message.content_type;
        }

        const options: RequestInit = {
            method: message.method.toUpperCase(),
            headers: headers,
        };

        if (message.method.toLowerCase() !== 'get' && message.body) {
            if (message.content_type && message.content_type.includes('application/json')) {
                options.body = JSON.stringify(message.body);
            } else {
                options.body = message.body;
            }
        }

        (async () => {
            try {
                const response = await fetch(url, options);
                let data: any;
                try {
                    data = await response.json();
                } catch (e) {
                    data = null;
                }

                if (!response.ok) {
                    const detail = data?.detail || 'Unknown';
                    if (message.on_response_error) {
                        message.on_response_error(detail);
                    }
                } else {
                    if (message.on_response_ok) {
                        message.on_response_ok(data);
                    }
                }
            } catch (error: any) {
                if (message.on_response_error) {
                    message.on_response_error(error.message || 'Unknown error');
                }
            }
        })();
    }
}
