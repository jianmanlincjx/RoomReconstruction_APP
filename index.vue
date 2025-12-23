<template>
	<view class="container">
		<view class="nav-bar">
			<text class="nav-title">室内空间三维重建</text>
		</view>

		<view class="content">
			<view class="guide-box" v-if="!hasResult">
				<text class="guide-title">📷 拍摄指引</text>
				<text class="guide-text">请在房间中心拍摄多张不同角度的照片，覆盖墙角、门窗等关键区域。</text>
			</view>

			<view class="section-title">
				已选视角 ({{displayList.length}}/20)
				<text v-if="loading" class="process-tip">{{uploadProgress}}</text>
			</view>
			
			<view class="image-grid">
				<view class="image-item" v-for="(img, index) in displayList" :key="index">
					<image :src="img" mode="aspectFill" class="thumb" @click="previewImage(img)"></image>
					<view class="delete-btn" @click="deleteImage(index)">×</view>
				</view>
				<view class="add-btn" @click="chooseImages" v-if="displayList.length < 20" :class="{ 'disabled': loading }">
					<text class="plus">+</text>
				</view>
			</view>

			<view class="action-area">
				<button v-if="!hasResult && !errorInfo.show" class="submit-btn" :loading="loading" :disabled="loading || displayList.length === 0" @click="startRebuild">
					{{ loading ? '🤖 ' + (uploadProgress || '云端计算中...') : '🚀 开始三维重建' }}
				</button>
				
				<view class="retry-controls" v-if="hasResult || errorInfo.show">
					<button class="clear-btn" @click="clearAll">✨ 开始新的重建</button>
				</view>
				
				<view class="error-box" v-if="errorInfo.show">
					<text class="error-title">发生错误</text>
					<text class="error-message">{{ errorInfo.message }}</text>
				</view>
			</view>

			<view class="result-container" v-if="hasResult">
				
				<view class="tabs">
					<view class="tab-item" :class="{ active: currentTab === 0 }" @click="currentTab = 0">
						<text>2D 平面图</text>
						<view class="tab-line"></view>
					</view>
					<view class="tab-item" :class="{ active: currentTab === 1 }" @click="currentTab = 1">
						<text>3D 结构图</text>
						<view class="tab-line"></view>
					</view>
					<view class="tab-item" :class="{ active: currentTab === 2 }" @click="currentTab = 2">
						<text>结构化参数</text>
						<view class="tab-line"></view>
					</view>
				</view>

				<view class="tab-content">
					
					<view v-if="currentTab === 0" class="tab-pane fade-in">
						<view class="card">
							<view class="card-header">
								<text class="tag blue">2D</text>
								<text>标准俯视结构图</text>
							</view>
							<image :src="result2d" mode="widthFix" class="result-img" @click="previewResult(result2d)"></image>
							<text class="hint">点击图片放大查看布局细节</text>
						</view>
					</view>

					<view v-if="currentTab === 1" class="tab-pane fade-in">
						<view class="card">
							<view class="card-header">
								<text class="tag">3D</text>
								<text>空间点云/网格视图</text>
							</view>
							<image :src="result3d" mode="widthFix" class="result-img" @click="previewResult(result3d)"></image>
							<text class="hint">点击图片放大查看三维结构</text>
						</view>
					</view>

					<view v-if="currentTab === 2" class="tab-pane fade-in">
						
						<view class="data-section">
							<text class="section-header">📊 空间概览</text>
							<view class="data-table summary-table">
								<view class="tr">
									<view class="th">估算面积</view>
									<view class="th">墙体总数</view>
									<view class="th">门 / 窗</view>
								</view>
								<view class="tr">
									<view class="td highlight">{{stats.area_sqm || 0}} ㎡</view>
									<view class="td">{{stats.counts.walls || 0}} 面</view>
									<view class="td">{{stats.counts.doors || 0}} / {{stats.counts.windows || 0}}</view>
								</view>
							</view>
						</view>

						<view class="data-section">
							<view class="flex-row-between">
								<text class="section-header">📏 墙体几何明细</text>
								<text class="sub-text">总周长: {{stats.lengths.total_wall_length}}m</text>
							</view>
							
							<view class="data-table detail-table">
								<view class="tr head">
									<view class="th col-id">编号</view>
									<view class="th col-type">类型</view>
									<view class="th col-val">几何数据 (长度)</view>
								</view>
								
								<view class="tr" v-for="(len, idx) in stats.lengths.wall_details" :key="'w'+idx">
									<view class="td col-id">W-{{idx + 1}}</view>
									<view class="td col-type"><text class="badge wall">墙体</text></view>
									<view class="td col-val">{{len}} 米</view>
								</view>

								<view class="tr" v-if="stats.counts.doors > 0">
									<view class="td col-id">D-ALL</view>
									<view class="td col-type"><text class="badge door">门</text></view>
									<view class="td col-val">共 {{stats.counts.doors}} 扇 (尺寸聚合)</view>
								</view>
								<view class="tr" v-if="stats.counts.windows > 0">
									<view class="td col-id">WIN-ALL</view>
									<view class="td col-type"><text class="badge window">窗</text></view>
									<view class="td col-val">共 {{stats.counts.windows}} 扇 (尺寸聚合)</view>
								</view>
								
								<view class="tr" v-if="!stats.lengths.wall_details || stats.lengths.wall_details.length === 0">
									<view class="td full-width">暂无详细几何数据</view>
								</view>
							</view>
						</view>
						
						<view class="api-info">
							<text>数据生成耗时: {{inferenceTime}}s</text>
						</view>

					</view>
				</view>
			</view>
			
		</view>
	</view>
</template>

<script>
	export default {
		data() {
			return {
				// 替换为你的实际接口地址
				apiUrl: 'https://dudley-undebased-tisa.ngrok-free.dev/predict_base64',
				
				displayList: [],
				loading: false,
				uploadProgress: '',
				
				// 结果数据
				result2d: '',
				result3d: '',
				stats: null,
				inferenceTime: 0,
				
				// 状态控制
				errorInfo: { show: false, message: '' },
				currentTab: 0, // 当前选中的 Tab：0=2D, 1=3D, 2=Data
			}
		},
		computed: {
			// 辅助判断是否有结果，用于控制显隐
			hasResult() {
				return !!(this.result2d || this.result3d || this.stats);
			}
		},
		methods: {
			async chooseImages() {
				if (this.loading) return;
				try {
					const res = await uni.chooseImage({
						count: 20 - this.displayList.length,
						sizeType: ['original'], // 建议使用原图以保证重建质量
						sourceType: ['album', 'camera'],
					});
					this.displayList = this.displayList.concat(res.tempFilePaths);
				} catch (e) {
					console.error(e);
				}
			},

			deleteImage(index) {
				this.displayList.splice(index, 1);
			},
			
			clearAll() {
				this.displayList = [];
				this.result2d = '';
				this.result3d = '';
				this.stats = null;
				this.inferenceTime = 0;
				this.errorInfo = { show: false, message: '' };
				this.loading = false;
				this.uploadProgress = '';
				this.currentTab = 0; // 重置 Tab
			},
			
			async startRebuild() {
				if (this.displayList.length === 0) {
					uni.showToast({ title: '请先选择图片', icon: 'none' });
					return;
				}
				this.loading = true;
				this.result2d = '';
				this.result3d = '';
				this.stats = null;
				this.errorInfo = { show: false, message: '' };
				
				let base64List = [];
				try {
					for (let i = 0; i < this.displayList.length; i++) {
						const path = this.displayList[i];
						this.uploadProgress = `本地处理: ${i + 1}/${this.displayList.length} 张...`;
						const base64 = await this.processImage(path);
						base64List.push(base64);
					}
				} catch (error) {
					this.loading = false;
					this.uploadProgress = '';
					this.errorInfo = { show: true, message: '本地图片处理失败: ' + error };
					return;
				}
				
				this.uploadProgress = '云端 AI 建模中...';
				this.sendRequest(base64List);
			},
			
			processImage(path) {
				return new Promise((resolve, reject) => {
					// 压缩图片以减少上传带宽，同时保证尺寸足够
					uni.compressImage({
						src: path,
						quality: 80, 
						targetWidth: 1024, // 稍微降低分辨率加快速度
						targetHeight: 1024,
						success: (res) => {
							let tempPath = res.tempFilePath;
							// #ifdef APP-PLUS
							if (tempPath.indexOf('_doc') === 0 || tempPath.indexOf('/') === 0) {
								tempPath = 'file://' + plus.io.convertLocalFileSystemURL(tempPath);
							}
							plus.io.resolveLocalFileSystemURL(tempPath, (entry) => {
								entry.file((file) => {
									var fileReader = new plus.io.FileReader();
									fileReader.onloadend = (e) => { resolve(e.target.result); };
									fileReader.readAsDataURL(file);
								}, (err) => { reject(err); });
							}, (err) => { reject(err); });
							// #endif
							
							// #ifndef APP-PLUS
							uni.getFileSystemManager().readFile({
								filePath: tempPath,
								encoding: 'base64',
								success: (data) => { resolve('data:image/jpeg;base64,' + data.data); },
								fail: (err) => { reject(err); }
							});
							// #endif
						},
						fail: (err) => { reject(err); }
					});
				});
			},

			sendRequest(images) {
				uni.request({
					url: this.apiUrl,
					method: 'POST',
					header: { 
						'content-type': 'application/json',
						'ngrok-skip-browser-warning': 'true'
					},
					data: {
						room_type: 'bedroom',
						return_2d: true,
						return_3d: true,
						images: images
					},
					timeout: 180000, // 3分钟超时
					success: (res) => {
						if (res.statusCode === 200 && res.data && res.data.status === 'success') {
							const data = res.data;
							this.inferenceTime = data.inference_time;
							this.stats = data.statistics; 
							if (data.visualization_2d) this.result2d = 'data:image/png;base64,' + data.visualization_2d;
							if (data.visualization_3d) this.result3d = 'data:image/png;base64,' + data.visualization_3d;
							
							// 自动跳转到结果区域
							this.$nextTick(() => {
								// 默认先看 2D
								this.currentTab = 0; 
								uni.pageScrollTo({ scrollTop: 400, duration: 300 });
							});
						} else {
							this.errorInfo = { show: true, message: '服务器分析失败，请检查图片或重试' };
						}
					},
					fail: (err) => {
						this.errorInfo = { show: true, message: '网络连接失败，请检查服务器地址' };
					},
					complete: () => {
						this.loading = false;
						this.uploadProgress = '';
					}
				});
			},
			
			previewImage(url) { if(url) uni.previewImage({ current: url, urls: this.displayList }); },
			previewResult(url) { if(url) uni.previewImage({ urls: [url] }); },
		}
	}
</script>

<style>
	/* 基础布局 */
	.container { background-color: #F7F8FA; min-height: 100vh; padding-bottom: 50px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
	.nav-bar { background: #fff; padding-top: var(--status-bar-height); padding-bottom: 12px; padding-left: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100; }
	.nav-title { font-size: 18px; font-weight: 600; color: #1a1a1a; }
	.content { padding: 15px; }
	
	/* 引导与输入 */
	.guide-box { background: #EBF5FF; padding: 12px 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #007AFF; }
	.guide-title { font-weight: 700; color: #007AFF; font-size: 15px; margin-bottom: 4px; display: block; }
	.guide-text { color: #505050; font-size: 13px; line-height: 1.4; }
	
	.section-title { font-size: 15px; font-weight: 600; margin-bottom: 12px; color: #333; display: flex; justify-content: space-between; align-items: center; }
	.process-tip { font-size: 12px; color: #FF9800; }
	
	/* 图片网格 */
	.image-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
	.image-item { width: calc(25% - 8px); aspect-ratio: 1; position: relative; border-radius: 6px; overflow: hidden; background: #eee; }
	.thumb { width: 100%; height: 100%; }
	.delete-btn { position: absolute; top: 0; right: 0; background: rgba(0,0,0,0.6); color: #fff; width: 20px; height: 20px; text-align: center; line-height: 18px; border-bottom-left-radius: 6px; z-index: 10; font-size: 14px; }
	.add-btn { width: calc(25% - 8px); aspect-ratio: 1; background: #fff; border: 1px dashed #ccc; border-radius: 6px; display: flex; align-items: center; justify-content: center; }
	.add-btn.disabled { background-color: #f5f5f5; border-color: #eee; }
	.plus { font-size: 30px; color: #ccc; }
	
	/* 按钮 */
	.action-area { margin-bottom: 20px; }
	.submit-btn { background: linear-gradient(90deg, #007AFF, #00C6FF); color: #fff; border-radius: 25px; font-size: 16px; font-weight: 600; box-shadow: 0 4px 12px rgba(0,122,255,0.3); border: none; }
	.clear-btn { background: #fff; color: #007AFF; border: 1px solid #007AFF; border-radius: 25px; font-size: 16px; font-weight: 600; margin-top: 15px; }
	
	/* 错误提示 */
	.error-box { background-color: #FFF0F0; border: 1px solid #FFC0C0; padding: 12px; border-radius: 8px; margin-top: 15px; }
	.error-title { font-weight: bold; color: #D32F2F; font-size: 14px; display: block; margin-bottom: 4px; }
	.error-message { color: #666; font-size: 13px; }

	/* ================= Tabs 样式 (新) ================= */
	.result-container { margin-top: 10px; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.05); min-height: 400px; }
	
	.tabs { display: flex; border-bottom: 1px solid #eee; background: #fff; }
	.tab-item { flex: 1; text-align: center; padding: 15px 0; font-size: 14px; color: #666; position: relative; transition: all 0.3s; }
	.tab-item.active { color: #007AFF; font-weight: bold; font-size: 15px; }
	.tab-line { position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 0; height: 3px; background: #007AFF; border-radius: 2px; transition: width 0.3s; }
	.tab-item.active .tab-line { width: 40%; }
	
	.tab-content { padding: 20px; background: #fff; }
	.tab-pane { width: 100%; }
	.fade-in { animation: fadeIn 0.4s ease-out; }

	/* 卡片与图片结果 */
	.card-header { display: flex; align-items: center; margin-bottom: 12px; }
	.tag { background: #333; color: #fff; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right: 8px; font-weight: 600; }
	.tag.blue { background: #007AFF; }
	.result-img { width: 100%; border-radius: 8px; border: 1px solid #eee; }
	.hint { display: block; text-align: center; font-size: 12px; color: #999; margin-top: 10px; }

	/* ================= 表格样式 (新) ================= */
	.section-header { font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px; display: block; }
	.sub-text { font-size: 12px; color: #888; }
	.flex-row-between { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
	.data-section { margin-bottom: 25px; }
	
	.data-table { border: 1px solid #EBEEF5; border-radius: 8px; overflow: hidden; }
	.tr { display: flex; border-bottom: 1px solid #EBEEF5; }
	.tr:last-child { border-bottom: none; }
	.tr.head { background-color: #F5F7FA; }
	
	.th { padding: 10px 8px; font-size: 12px; color: #606266; font-weight: bold; text-align: center; flex: 1; }
	.td { padding: 12px 8px; font-size: 13px; color: #303133; text-align: center; flex: 1; display: flex; align-items: center; justify-content: center; }
	
	/* 汇总表特殊样式 */
	.summary-table .td { font-size: 14px; }
	.summary-table .highlight { color: #007AFF; font-weight: bold; font-size: 16px; }
	
	/* 明细表列宽控制 */
	.col-id { flex: 0.3; color: #909399; font-family: monospace; }
	.col-type { flex: 0.4; }
	.col-val { flex: 0.8; text-align: right; justify-content: flex-end; padding-right: 15px; font-weight: 500; }
	.full-width { flex: 1; color: #999; padding: 20px; }
	
	/* 徽章样式 */
	.badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; color: #fff; }
	.badge.wall { background-color: #909399; }
	.badge.door { background-color: #E6A23C; }
	.badge.window { background-color: #67C23A; }
	
	.api-info { text-align: right; font-size: 11px; color: #ccc; margin-top: 10px; }
	
	@keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
</style>