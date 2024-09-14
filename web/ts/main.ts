// 定义接口类型
import {send_http} from "./common";
import {get_authorization} from "./common";
get_authorization().then((response: { user_id: string; access_token: string } | null) => {
        if (!response || !response.user_id || !response.access_token) {
            console.log('Authorization not found');
            login();
        }
    }).catch(reason => {
        console.log(reason);
        login();
});
// 定义 login 函数，带有 fetch 请求
function login(): void {
    const on_response_ok=(response_data: any) =>{
                            if (response_data && response_data.access_token && response_data.user_id) {
                                const access_token = response_data.access_token;
                                const user_id = response_data.user_id;
                                localStorage.setItem('access_token', access_token);
                                localStorage.setItem('user_id', user_id);
                            } else {
                                console.error({ status: 'error', message: 'Invalid login response' });
                            }
                         }
    const on_response_error= (error:string) => {
                    console.error('Error:', error);
               }

    send_http({
         method:'post',
         url:'http://0.0.0.0:8000/login',
         params:undefined,
         body:new URLSearchParams({
             username: 'testuser',
             password: 'testpassword',
         }),
         token:localStorage.getItem("access_token") as string,
         content_type:'application/x-www-form-urlencoded',
         on_response_ok:(response_data: any)=>on_response_ok(response_data),
         on_response_error:(error:string)=>on_response_error(error)
        }
    )
}