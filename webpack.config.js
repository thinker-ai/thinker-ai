const path = require('path');

module.exports = {
  mode: 'development',  // 设置为开发模式
  entry: {
    main: './web/ts/main.ts',  // 前端 TypeScript 的入口文件
    dynamic_ui:'./web/ts/design/dynamic-ui.ts',
    floating_panel:'./web/ts/design/floating-panel.ts',
    solution:'./web/ts/design/one/solution.ts',
    common:'./web/ts/common.ts',
    request_sender_front:'/web/ts/request_sender_front.ts',
    request_sender_worker:'/web/ts/request_sender_worker.ts',
    request_sender_background:'/web/ts/request_sender_background.ts',
    web_socket_front:'/web/ts/web_socket_front.ts',
    web_socket_background:'/web/ts/web_socket_background.ts',
    web_socket_worker:'/web/ts/web_socket_worker.ts',
  },
  output: {
    filename: '[name].js',  // 使用入口名称作为输出文件名
    path: path.resolve(__dirname, './web/js'),  // 输出路径为 web/js 目录
  },
  resolve: {
    extensions: ['.ts', '.js'],  // 允许解析 .ts 和 .js 文件
    modules: [path.resolve(__dirname, 'web/ts'), 'node_modules'],  // 指定模块解析目录为 web/ts 和 node_modules
  },
  module: {
    rules: [
      {
        test: /\.ts$/,  // 匹配所有以 .ts 结尾的文件
        use: 'ts-loader',  // 使用 ts-loader 将 TypeScript 转换为 JavaScript
        exclude: /node_modules/,  // 排除 node_modules 目录中的文件
      },
      {
        enforce: 'pre',  // 确保这个 loader 在其他 loader 之前运行
        test: /\.js$/,  // 匹配所有以 .js 结尾的文件
        loader: 'source-map-loader',  // 使用 source-map-loader 加载 .js 文件的 source maps
        exclude: /node_modules/,  // 排除 node_modules 中的文件
      },
    ],
  },
  devtool: 'source-map',  // 开启 source maps 以便于调试
  watch: true,  // 监控文件变化并自动打包
};