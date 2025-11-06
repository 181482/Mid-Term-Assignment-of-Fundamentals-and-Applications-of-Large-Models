import subprocess
import sys
import logging
import argparse
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_command(cmd, description):
    """运行shell命令并记录日志"""
    logging.info("="*80)
    logging.info(f"开始: {description}")
    logging.info(f"命令: {cmd}")
    logging.info("="*80)
    
    start_time = time.time()
    
    try:
        # 运行命令并实时输出
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程结束
        return_code = process.wait()
        
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            logging.info(f"✓ {description} 完成! 耗时: {elapsed_time/60:.2f} 分钟")
            return True
        else:
            logging.error(f"✗ {description} 失败! 返回码: {return_code}")
            return False
            
    except Exception as e:
        logging.error(f"✗ 执行 {description} 时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='运行完整的Transformer训练和评估流程')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='使用的设备 (cuda或cpu)')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练,直接运行消融实验')
    parser.add_argument('--skip-ablation', action='store_true',
                       help='跳过消融实验')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='跳过结果分析')
    parser.add_argument('--skip-samples', action='store_true',
                       help='跳过翻译样例生成')
    args = parser.parse_args()
    
    device = args.device
    total_start_time = time.time()
    
    logging.info("\n" + "="*80)
    logging.info("Transformer 完整流程启动")
    logging.info(f"设备: {device}")
    logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80 + "\n")
    
    # # 步骤1: 数据检查
    # logging.info("\n📊 步骤 1: 数据质量检查")
    # if not run_command(
    #     "python src/inspect_data.py",
    #     "数据质量检查"
    # ):
    #     logging.error("数据检查失败,请检查数据集!")
    #     sys.exit(1)
    
    # 步骤2: 主训练
    if not args.skip_train:
        logging.info("\n🚀 步骤 2: 主模型训练")
        if not run_command(
            f"bash scripts/run.sh --mode train --device {device}",
            "主模型训练"
        ):
            logging.error("主训练失败!")
            if input("是否继续执行后续步骤? (y/n): ").lower() != 'y':
                sys.exit(1)
    else:
        logging.info("\n⏭️  跳过主模型训练")
    
    # 步骤5: 生成翻译样例
    if not args.skip_samples:
        logging.info("\n💬 步骤 5: 生成翻译样例")
        if not run_command(
            "python src/generate_samples.py",
            "翻译样例生成"
        ):
            logging.warning("翻译样例生成失败")
    else:
        logging.info("\n⏭️  跳过翻译样例生成")


    # 步骤3: 消融实验
    if not args.skip_ablation:
        logging.info("\n🔬 步骤 3: 消融实验")
        if not run_command(
            f"bash scripts/run.sh --mode ablation --device {device}",
            "消融实验"
        ):
            logging.warning("消融实验失败,继续后续步骤...")
    else:
        logging.info("\n⏭️  跳过消融实验")
    
    # 步骤4: 结果分析
    if not args.skip_analysis:
        logging.info("\n📈 步骤 4: 结果分析")
        if not run_command(
            "python src/analyze_ablation.py",
            "结果分析"
        ):
            logging.warning("结果分析失败,可能是没有足够的实验数据")
    else:
        logging.info("\n⏭️  跳过结果分析")
    

    
    # 步骤6: 生成可视化图表
    logging.info("\n📊 步骤 6: 生成可视化图表")
    if not run_command(
        "python src/visualize_results.py",
        "可视化生成"
    ):
        logging.warning("可视化生成失败")
    
    # 总结
    total_elapsed_time = time.time() - total_start_time
    
    logging.info("\n" + "="*80)
    logging.info("✨ 完整流程执行完毕!")
    logging.info(f"总耗时: {total_elapsed_time/3600:.2f} 小时 ({total_elapsed_time/60:.2f} 分钟)")
    logging.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)
    
    # 显示结果位置
    logging.info("\n📁 结果文件位置:")
    logging.info("  - 模型checkpoint: checkpoints/")
    logging.info("  - 训练曲线: results/training_curves.png")
    logging.info("  - 可视化图表: results/visualizations/")
    logging.info("  - 消融实验: results/ablation/")
    logging.info("  - 翻译样例: results/translation_samples.md")
    logging.info("  - WandB日志: https://wandb.ai")
    logging.info("\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\n⚠️  流程被用户中断")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n\n❌ 流程执行失败: {str(e)}")
        sys.exit(1)
