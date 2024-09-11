// 导入 axios 类型
import axios, { AxiosRequestConfig, AxiosResponse} from 'axios';
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
    makeRequest(message:RequestMessage): void;
}
// 定义 BaseRequestSender 类
export class RequestSender implements RequestSenderInterface {
    public makeRequest(message:RequestMessage): void {
        let axiosInstance = axios.create();
        const config: AxiosRequestConfig = {
            method:message.method,
            url:message.url,
            params:message.params,
            data: message.body,
            headers: {
                 'Content-Type': message.content_type,
            },
        };

        // 如果需要 token，则在请求头中添加 Authorization
        if (message.token) {
            config.headers = {
                ...config.headers, // 保留已有 headers
                Authorization: `Bearer ${message.token}`
            };
        }

        axiosInstance(config)
            .then((response: AxiosResponse<any>) => {
                if (message.on_response_ok) {
                    message.on_response_ok(response.data);
                }
            })
            .catch((error: any) => {
                const detail = error.response?.data.detail || 'Unknown';
                if (message.on_response_error) {
                    message.on_response_error(detail);
                }
            });
    }
}