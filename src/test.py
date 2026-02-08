import sys
sys.path.insert(0, '.')

from train import FullTrainer
from datetime import datetime
from pathlib import Path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained MixRAGRec model")
    
    # Experiment name (for locating model directory)
    parser.add_argument('--experiment', type=str, default=None, 
                        help='Experiment name (e.g., MixRAGRec_ml1m_llama-8b). If not specified, uses config default.')
    
    # Model selection: --epoch or --best
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--epoch', type=int, help='Epoch number to load')
    model_group.add_argument('--best', action='store_true', help='Load the best model (saved_models/{experiment}/best/)')
    
    parser.add_argument('--num-samples', type=int, default=None, help='Number of test samples (default: all 1000)')
    parser.add_argument('--show-details', action='store_true', help='Show detailed outputs')
    parser.add_argument('--detail-samples', type=int, default=3, help='Number of samples to show details for')
    parser.add_argument('--output-file', type=str, default=None, help='Output file path for saving results (default: results/test_results_TIMESTAMP.txt)')
    
    args = parser.parse_args()
    
    # Determine model to load
    if args.best:
        args.epoch = 'best'  # Use 'best' as special identifier
    
    print("="*80)
    print("MixRAGRec Model Testing")
    print("="*80)
    
    # Initialize trainer
    trainer = FullTrainer()
    
    # If experiment name is specified, override default save directory
    if args.experiment:
        trainer.experiment_name = args.experiment
        trainer.model_save_dir = Path("saved_models") / args.experiment
        print(f"  Using experiment: {args.experiment}")
    
    # Load model with verbose metadata display
    epoch_str = "best" if args.best else f"epoch {args.epoch}"
    print(f"\nLoading model from {trainer.model_save_dir}/{epoch_str}...")
    success = trainer.load_models(args.epoch, verbose=True)
    
    if not success:
        print("Failed to load model. Exiting.")
        return 1
    
    # Get model metadata for output
    if args.best:
        load_path = trainer.model_save_dir / "best"
    else:
        load_path = trainer.model_save_dir / f"epoch_{args.epoch}"
    
    model_metadata = {}
    try:
        import torch
        training_state = torch.load(load_path / "training_state.pt", weights_only=False)
        model_metadata = {
            'experiment_name': training_state.get('experiment_name', 'N/A'),
            'timestamp': training_state.get('timestamp', 'N/A'),
            'accuracy': training_state.get('accuracy', 0),
            'top3_accuracy': training_state.get('top3_accuracy', 0),
            'top5_accuracy': training_state.get('top5_accuracy', 0),
            'dataset': training_state.get('dataset', {}),
            'training_params': training_state.get('training_params', {}),
            'marl_params': training_state.get('marl_params', {}),
            'reward_params': training_state.get('reward_params', {}),
        }
    except:
        pass
    
    # Test on test set
    num_samples = args.num_samples if args.num_samples else len(trainer.test_data)
    
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)
    print(f"\nEvaluating on {num_samples} test samples...")
    
    # ============================================================================
    # Show detailed outputs for first N samples if enabled
    # ============================================================================
    if args.show_details:
        print("\n" + "="*80)
        print("DETAILED OUTPUTS")
        print("="*80)
        
        # Only show details for first N samples
        detail_count = min(args.detail_samples, num_samples)
        
        for idx in range(detail_count):
            sample = trainer.test_data[idx]
            
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{detail_count}")
            print(f"{'='*80}")
            
            # Show input
            print(f"\n【Input - Watching History】")
            history = sample['input']
            print(f"{history[:200]}..." if len(history) > 200 else history)
            
            print(f"\n【Options】")
            options_str = sample['questions']
            print(f"{options_str[:300]}..." if len(options_str) > 300 else options_str)
            
            print(f"\n【Ground Truth】")
            print(f"Correct Answer: {sample['output']}")
            
            # Preprocess sample
            preprocessed = trainer.preprocess_sample(sample)
            
            # Execute pipeline
            print(f"\n【Pipeline Execution】")
            result = trainer.agent_manager.execute_pipeline(
                user_query=preprocessed['task_prompt'],
                user_context={'retrieval_query': preprocessed['retrieval_query'], 
                             'options': preprocessed['options']},
                conversation_history=[],
                training=False
            )
            # Use constrained decoding result
            if result.get('pipeline_success'):
                predicted_answer = result.get('predicted_option')
                if predicted_answer is None:
                    # Fallback to first option if constrained decoding fails
                    predicted_answer = list(preprocessed['options'].keys())[0] if preprocessed['options'] else 'A'
                    print(f"  ⚠ Constrained decoding failed, using default: {predicted_answer}")
                result['predicted_answer'] = predicted_answer
            
            if result.get('pipeline_success'):
                # Step 1: Retrieval
                print(f"\n--- Step 1: Retrieval ---")
                expert_id = result.get('selected_retriever_id', 'Unknown')
                expert_names = {1: "DirectGenerator", 2: "TripleRetriever", 
                               3: "SubgraphRetriever", 4: "ConnectedGraphRetriever"}
                expert_name = expert_names.get(expert_id, f"Expert {expert_id}")
                print(f"Selected Expert: {expert_name} (ID: {expert_id})")
                
                # Step 2: Knowledge Alignment
                print(f"\n--- Step 2: Knowledge Alignment ---")
                refined = result.get('refined_knowledge', '')
                print(f"Refined Knowledge Length: {len(refined)} chars")
                if refined:
                    print(f"\nRefined Knowledge (preview):")
                    print(refined[:500] + "..." if len(refined) > 500 else refined)
                print(f"\nAlignment Confidence: {result.get('refinement_confidence', 0):.3f}")
                
                # Step 3: Recommendations
                print(f"\n--- Step 4: Final Recommendations ---")
                recommendations = result.get('final_recommendations', '')
                print(f"Recommendations Length: {len(recommendations)} chars")
                if recommendations:
                    print(f"\n【Recommendations Content】:")
                    print(recommendations)
                else:
                    print("(No recommendations generated)")
                
                # Prediction vs Ground Truth
                print(f"\n--- Prediction vs Ground Truth ---")
                predicted = result.get('predicted_answer', 'N/A')
                correct = sample['output']
                is_correct = (predicted == correct)
                print(f"Predicted Answer: {predicted}")
                print(f"Ground Truth: {correct}")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                
                # Quality Metrics
                print(f"\n--- Quality Metrics ---")
                print(f"Generation Confidence: {result.get('generation_confidence', 0):.3f}")
                print(f"Diversity Score: {result.get('diversity_score', 0):.3f}")
                print(f"Quality Score: {result.get('quality_score', 0):.3f}")
                
            else:
                print(f"\n✗ Pipeline Failed: {result.get('error_message')}")
            
            print(f"\n{'='*80}\n")
        
        print("\nContinuing with full evaluation...\n")
    
    # ============================================================================
    # Standard evaluation (all samples)
    # ============================================================================
    from tqdm import tqdm
    
    correct = 0
    top3_correct = 0
    top5_correct = 0
    total_reward = 0.0
    expert_usage = {}
    total_evaluated = 0
    
    for sample in tqdm(trainer.test_data[:num_samples], desc="Evaluating"):
        preprocessed = trainer.preprocess_sample(sample)
        # Test mode, don't affect training state
        result = trainer.agent_manager.execute_pipeline(
            user_query=preprocessed['task_prompt'],
            user_context={'retrieval_query': preprocessed['retrieval_query'], 
                         'options': preprocessed['options']},
            conversation_history=[],
            training=False
        )
        
        if result.get('pipeline_success'):
            total_evaluated += 1
            
            # Use constrained decoding result
            predicted = result.get('predicted_option')
            if predicted is None:
                predicted = list(preprocessed['options'].keys())[0] if preprocessed['options'] else 'A'
            
            ground_truth = preprocessed['correct_answer']
            is_correct = (predicted == ground_truth)
            
            if is_correct:
                correct += 1
                # Top-1 correct → Top-3 and Top-5 necessarily correct
                top3_correct += 1
                top5_correct += 1
            else:
                # Check if ground_truth is in Top-K
                option_distribution = result.get('option_distribution')
                if option_distribution and len(option_distribution) > 0:
                    sorted_options = sorted(option_distribution.items(), key=lambda x: x[1], reverse=True)
                    top3_options = [opt for opt, _ in sorted_options[:3]]
                    top5_options = [opt for opt, _ in sorted_options[:5]]
                    if ground_truth in top3_options:
                        top3_correct += 1
                    if ground_truth in top5_options:
                        top5_correct += 1
            
            # Track expert usage
            retriever_id = result.get('selected_retriever_id')
            if retriever_id is not None:
                expert_usage[retriever_id] = expert_usage.get(retriever_id, 0) + 1
    
    accuracy = correct / total_evaluated if total_evaluated > 0 else 0
    top3_accuracy = top3_correct / total_evaluated if total_evaluated > 0 else 0
    top5_accuracy = top5_correct / total_evaluated if total_evaluated > 0 else 0
    avg_reward = 0.0  # Placeholder for compatibility
    
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)
    print(f"Accuracy (Top-1): {accuracy:.3f} ({correct}/{total_evaluated})")
    print(f"Top-3 Accuracy:   {top3_accuracy:.3f} ({top3_correct}/{total_evaluated})")
    print(f"Top-5 Accuracy:   {top5_accuracy:.3f} ({top5_correct}/{total_evaluated})")
    
    # Print expert usage
    print(f"\nExpert Usage Distribution:")
    expert_names = {1: "Direct", 2: "Triple", 3: "Subgraph", 4: "Connected"}
    total_usage = sum(expert_usage.values())
    for expert_id in sorted(expert_usage.keys()):
        count = expert_usage[expert_id]
        percentage = (count / total_usage * 100) if total_usage > 0 else 0
        name = expert_names.get(expert_id, f"Expert{expert_id}")
        print(f"  Expert {expert_id} ({name}): {count} ({percentage:.1f}%)")
    
    print("="*80)
    
    # ============================================================================
    # Save test results to txt file
    # ============================================================================
    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Default save to results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_label = "best" if args.best else f"epoch{args.epoch}"
        output_path = results_dir / f"test_results_{epoch_label}_{timestamp}.txt"
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write results file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MixRAGRec Model Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic info
        f.write("【Test Configuration】\n")
        f.write(f"  Model: {epoch_label}\n")
        f.write(f"  Test Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Num Samples: {num_samples}\n")
        f.write(f"  Constrained Decoding: Enabled\n")
        f.write("\n")
        
        # Model training metadata
        if model_metadata:
            f.write("【Model Training Metadata】\n")
            f.write(f"  Trained at: {model_metadata.get('timestamp', 'N/A')}\n")
            
            dataset_info = model_metadata.get('dataset', {})
            if dataset_info:
                f.write(f"  Dataset: {dataset_info.get('name', 'N/A')}\n")
                f.write(f"  Train/Test Size: {dataset_info.get('train_size', 'N/A')}/{dataset_info.get('test_size', 'N/A')}\n")
            
            training_params = model_metadata.get('training_params', {})
            if training_params:
                f.write(f"  Samples/Epoch: {training_params.get('samples_per_epoch', 'N/A')}\n")
                f.write(f"  Total Epochs: {training_params.get('num_epochs', 'N/A')}\n")
                f.write(f"  Seed: {training_params.get('seed', 'N/A')}\n")
            
            marl_params = model_metadata.get('marl_params', {})
            if marl_params:
                policy_opt = marl_params.get('policy', {})
                pref_opt = marl_params.get('preference', {})
                f.write(f"  Policy Optimization LR: {policy_opt.get('learning_rate', 'N/A')}, γ: {policy_opt.get('gamma', 'N/A')}, ε: {policy_opt.get('clip_epsilon', 'N/A')}\n")
                f.write(f"  Preference Optimization β: {pref_opt.get('beta', 'N/A')}\n")
            
            reward_params = model_metadata.get('reward_params', {})
            if reward_params:
                f.write(f"  Reward: λ={reward_params.get('lambda_mig', 'N/A')}, η={reward_params.get('eta_cost', 'N/A')}\n")
            f.write("\n")
        
        # Main metrics
        f.write("【Main Metrics】\n")
        f.write(f"  Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)\n")
        f.write(f"  Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)\n")
        f.write(f"  Correct (Top-1): {correct}/{total_evaluated}\n")
        f.write(f"  Correct (Top-3): {top3_correct}/{total_evaluated}\n")
        f.write(f"  Correct (Top-5): {top5_correct}/{total_evaluated}\n")
        f.write("\n")
        
        # Expert usage distribution
        f.write("【Expert Usage Distribution】\n")
        for expert_id in sorted(expert_usage.keys()):
            count = expert_usage[expert_id]
            percentage = (count / total_usage * 100) if total_usage > 0 else 0
            name = expert_names.get(expert_id, f"Expert{expert_id}")
            f.write(f"  Expert {expert_id} ({name}): {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Calculate expert selection entropy (diversity metric)
        if total_usage > 0:
            import math
            entropy = 0.0
            for count in expert_usage.values():
                if count > 0:
                    p = count / total_usage
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(len(expert_usage)) if len(expert_usage) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            f.write("【Expert Selection Diversity】\n")
            f.write(f"  Entropy: {entropy:.3f}\n")
            f.write(f"  Normalized Entropy: {normalized_entropy:.3f} (0=single selection, 1=uniform distribution)\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Test results saved to: {output_path}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
