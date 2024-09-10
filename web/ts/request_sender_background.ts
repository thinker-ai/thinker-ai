// 导入 axios 类型
import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig,AxiosRequestHeaders} from 'axios';

// 扩展 InternalAxiosRequestConfig 类型以包含 useToken
interface CustomAxiosRequestConfig extends InternalAxiosRequestConfig<any> {
    useToken?: boolean;
}

function loadAxios(token: string | null = null): AxiosInstance {
    const axiosInstance = axios.create();

    axiosInstance.interceptors.request.use(
        async function (config: CustomAxiosRequestConfig): Promise<InternalAxiosRequestConfig<any>> {
            // 检查是否需要使用 token
            if (token && config.useToken) {
                // 检查是否有 headers，没有则创建 AxiosRequestHeaders 类型的 headers
                if (!config.headers) {
                    config.headers = {} as AxiosRequestHeaders;
                }
                config.headers.Authorization = 'Bearer ' + token;
            }
            return config; // 返回 InternalAxiosRequestConfig 类型
        },
        function (error: AxiosError) {
            return Promise.reject(error); // 错误处理保持不变
        }
    );
    return axiosInstance;
}

export type RequestMessage = {
        method: 'get' | 'post',
        url: string,
        params?: URLSearchParams,
        body?: any,
        useToken?: boolean,
        content_type?:string,
        on_response_ok?:(response_data: any) => void,
        on_response_error?:(error_status: number|string) => void,
};
// 定义请求配置的接口
export interface RequestSenderInterface {
    makeRequest(message:RequestMessage): void;
}
// 定义 BaseRequestSender 类
export class RequestSender implements RequestSenderInterface {
    private token: string | null;
    public constructor(token: string | null = null) {
        this.token = token;
    }
    public makeRequest(message:RequestMessage): void {
        let axiosInstance = loadAxios(this.token);
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
        if (message.useToken && this.token) {
            config.headers = {
                ...config.headers, // 保留已有 headers
                Authorization: `Bearer ${this.token}`
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