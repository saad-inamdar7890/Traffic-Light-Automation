"""
Configuration Management Module
===============================
Centralized configuration management for the traffic light automation system.
Handles all SUMO parameters, algorithm settings, file paths, and simulation parameters.

Features:
- Centralized configuration management
- Environment-specific settings
- Parameter validation
- Default configurations
- Configuration file loading/saving
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ControllerConfig:
    """Configuration for the adaptive traffic controller."""
    # Timing parameters
    min_phase_duration: int = 10
    max_phase_duration: int = 120
    yellow_duration: int = 4
    red_duration: int = 2
    
    # Adaptation parameters
    adaptation_interval: int = 15
    max_adjustment: int = 40
    pressure_threshold: float = 5.0
    early_transition_threshold: float = 2.0
    
    # Traffic pressure calculation
    vehicle_weight: float = 5.0
    waiting_weight: float = 2.0
    speed_weight: float = 1.0
    
    # Performance parameters
    enable_early_transitions: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class SimulationConfig:
    """Configuration for SUMO simulation parameters."""
    # Basic simulation parameters
    duration: int = 3600
    step_length: float = 1.0
    begin_time: int = 0
    
    # Network and route files
    network_file: str = "../demo.net.xml"
    route_file: str = "../demo.rou.xml"
    config_file: str = "../demo.sumocfg"
    
    # SUMO options
    enable_gui: bool = False
    no_warnings: bool = True
    no_step_log: bool = True
    seed: Optional[int] = None
    
    # Additional parameters
    default_speeddev: float = 0.1
    pedestrian_model: str = "none"
    collision_action: str = "warn"


@dataclass
class AnalysisConfig:
    """Configuration for traffic analysis and monitoring."""
    # Data collection
    collection_interval: int = 1
    report_interval: int = 60
    save_interval: int = 300
    
    # Performance metrics
    enable_real_time_analysis: bool = True
    enable_directional_analysis: bool = True
    
    # Export settings
    export_data: bool = True
    export_format: str = "json"
    
    # Visualization
    enable_real_time_visualization: bool = False
    plot_interval: int = 120
    figure_size: tuple = (12, 8)


@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting."""
    # Chart settings
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8-darkgrid"
    
    # Colors
    adaptive_color: str = "#2E8B57"
    normal_color: str = "#DC143C"
    background_color: str = "#F0F0F0"
    
    # Export settings
    save_plots: bool = True
    plot_format: str = "png"
    plot_quality: int = 95


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    # Base directories
    base_directory: str = "."
    src_directory: str = "src"
    results_directory: str = "results"
    temp_directory: str = "temp"
    
    # File patterns
    route_file_pattern: str = "route_{scenario}_{timestamp}.rou.xml"
    config_file_pattern: str = "config_{scenario}_{timestamp}.sumocfg"
    results_file_pattern: str = "results_{scenario}_{timestamp}.json"
    
    # Cleanup settings
    cleanup_temp_files: bool = True
    keep_last_n_files: int = 10


@dataclass
class SystemConfig:
    """Main system configuration combining all sub-configurations."""
    controller: ControllerConfig
    simulation: SimulationConfig
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    paths: PathConfig
    
    # System-wide settings
    debug_mode: bool = False
    verbose_logging: bool = False
    performance_monitoring: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        # Validate controller parameters
        if self.controller.min_phase_duration >= self.controller.max_phase_duration:
            raise ValueError("min_phase_duration must be less than max_phase_duration")
        
        if self.controller.adaptation_interval <= 0:
            raise ValueError("adaptation_interval must be positive")
        
        # Validate simulation parameters
        if self.simulation.duration <= 0:
            raise ValueError("simulation duration must be positive")
        
        if self.simulation.step_length <= 0:
            raise ValueError("step_length must be positive")
        
        # Validate analysis parameters
        if self.analysis.collection_interval <= 0:
            raise ValueError("collection_interval must be positive")
        
        # Validate paths
        if not self.paths.base_directory:
            raise ValueError("base_directory cannot be empty")


class ConfigManager:
    """
    Configuration manager for loading, saving, and managing system configurations.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file
        self._config = None
        
    def get_default_config(self) -> SystemConfig:
        """
        Get default system configuration.
        
        Returns:
            Default SystemConfig instance
        """
        return SystemConfig(
            controller=ControllerConfig(),
            simulation=SimulationConfig(),
            analysis=AnalysisConfig(),
            visualization=VisualizationConfig(),
            paths=PathConfig()
        )
    
    def load_config(self, config_file: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Loaded SystemConfig instance
        """
        if config_file:
            self.config_file = config_file
        
        if not self.config_file or not os.path.exists(self.config_file):
            print(f"⚙️  Configuration file not found, using defaults")
            return self.get_default_config()
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Create configuration objects from loaded data
            controller_config = ControllerConfig(**config_data.get('controller', {}))
            simulation_config = SimulationConfig(**config_data.get('simulation', {}))
            analysis_config = AnalysisConfig(**config_data.get('analysis', {}))
            visualization_config = VisualizationConfig(**config_data.get('visualization', {}))
            paths_config = PathConfig(**config_data.get('paths', {}))
            
            # System-wide settings
            system_settings = config_data.get('system', {})
            
            config = SystemConfig(
                controller=controller_config,
                simulation=simulation_config,
                analysis=analysis_config,
                visualization=visualization_config,
                paths=paths_config,
                debug_mode=system_settings.get('debug_mode', False),
                verbose_logging=system_settings.get('verbose_logging', False),
                performance_monitoring=system_settings.get('performance_monitoring', True)
            )
            
            print(f"✅ Configuration loaded from {self.config_file}")
            self._config = config
            return config
            
        except Exception as e:
            print(f"⚠️  Error loading configuration: {e}")
            print(f"   Using default configuration")
            return self.get_default_config()
    
    def save_config(self, config: SystemConfig, config_file: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: SystemConfig instance to save
            config_file: Path to configuration file
            
        Returns:
            True if saved successfully
        """
        if config_file:
            self.config_file = config_file
        
        if not self.config_file:
            self.config_file = "system_config.json"
        
        try:
            # Convert configuration to dictionary
            config_data = {
                'controller': asdict(config.controller),
                'simulation': asdict(config.simulation),
                'analysis': asdict(config.analysis),
                'visualization': asdict(config.visualization),
                'paths': asdict(config.paths),
                'system': {
                    'debug_mode': config.debug_mode,
                    'verbose_logging': config.verbose_logging,
                    'performance_monitoring': config.performance_monitoring
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"💾 Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error saving configuration: {e}")
            return False
    
    def update_config(self, **kwargs) -> SystemConfig:
        """
        Update configuration with new parameters.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            Updated SystemConfig instance
        """
        if not self._config:
            self._config = self.get_default_config()
        
        # Update configuration sections
        for section, params in kwargs.items():
            if hasattr(self._config, section):
                section_config = getattr(self._config, section)
                for param, value in params.items():
                    if hasattr(section_config, param):
                        setattr(section_config, param, value)
        
        return self._config
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get scenario-specific configuration.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Scenario configuration dictionary
        """
        scenario_configs = {
            'balanced': {
                'description': 'Balanced traffic in all directions',
                'duration': 3600,
                'report_interval': 60
            },
            'heavy_ns': {
                'description': 'Heavy North-South traffic',
                'duration': 1800,
                'report_interval': 30
            },
            'heavy_ew': {
                'description': 'Heavy East-West traffic',
                'duration': 1800,
                'report_interval': 30
            },
            'rush_hour': {
                'description': 'Rush hour simulation',
                'duration': 2700,
                'report_interval': 45
            }
        }
        
        return scenario_configs.get(scenario_name, scenario_configs['balanced'])
    
    def get_environment_config(self, environment: str = 'development') -> Dict[str, Any]:
        """
        Get environment-specific configuration overrides.
        
        Args:
            environment: Environment name (development, testing, production)
            
        Returns:
            Environment configuration dictionary
        """
        env_configs = {
            'development': {
                'simulation': {
                    'enable_gui': True,
                    'no_warnings': False,
                    'duration': 900
                },
                'analysis': {
                    'enable_real_time_visualization': True,
                    'report_interval': 30
                },
                'system': {
                    'debug_mode': True,
                    'verbose_logging': True
                }
            },
            'testing': {
                'simulation': {
                    'enable_gui': False,
                    'no_warnings': True,
                    'duration': 300
                },
                'analysis': {
                    'enable_real_time_visualization': False,
                    'report_interval': 60
                },
                'system': {
                    'debug_mode': False,
                    'verbose_logging': False
                }
            },
            'production': {
                'simulation': {
                    'enable_gui': False,
                    'no_warnings': True,
                    'duration': 3600
                },
                'analysis': {
                    'enable_real_time_visualization': False,
                    'report_interval': 300
                },
                'system': {
                    'debug_mode': False,
                    'verbose_logging': False
                }
            }
        }
        
        return env_configs.get(environment, {})
    
    def validate_paths(self, config: SystemConfig) -> bool:
        """
        Validate that all required paths exist or can be created.
        
        Args:
            config: SystemConfig instance to validate
            
        Returns:
            True if all paths are valid
        """
        try:
            paths = config.paths
            
            # Check base directory
            if not os.path.exists(paths.base_directory):
                os.makedirs(paths.base_directory, exist_ok=True)
            
            # Check/create results directory
            results_path = os.path.join(paths.base_directory, paths.results_directory)
            if not os.path.exists(results_path):
                os.makedirs(results_path, exist_ok=True)
            
            # Check/create temp directory
            temp_path = os.path.join(paths.base_directory, paths.temp_directory)
            if not os.path.exists(temp_path):
                os.makedirs(temp_path, exist_ok=True)
            
            # Check network file
            network_path = os.path.join(paths.base_directory, config.simulation.network_file)
            if not os.path.exists(network_path):
                print(f"⚠️  Network file not found: {network_path}")
                return False
            
            print("✅ All paths validated successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  Error validating paths: {e}")
            return False
    
    def print_config_summary(self, config: SystemConfig):
        """
        Print a formatted summary of the current configuration.
        
        Args:
            config: SystemConfig instance to summarize
        """
        print(f"\n⚙️  CONFIGURATION SUMMARY")
        print(f"{'='*50}")
        
        print(f"\n🧠 CONTROLLER:")
        print(f"   Phase Duration: {config.controller.min_phase_duration}-{config.controller.max_phase_duration}s")
        print(f"   Adaptation Interval: {config.controller.adaptation_interval}s")
        print(f"   Max Adjustment: {config.controller.max_adjustment}s")
        print(f"   Early Transitions: {'Yes' if config.controller.enable_early_transitions else 'No'}")
        
        print(f"\n🚗 SIMULATION:")
        print(f"   Duration: {config.simulation.duration}s ({config.simulation.duration/60:.1f} min)")
        print(f"   Network: {config.simulation.network_file}")
        print(f"   GUI: {'Enabled' if config.simulation.enable_gui else 'Disabled'}")
        print(f"   Step Length: {config.simulation.step_length}s")
        
        print(f"\n📊 ANALYSIS:")
        print(f"   Collection Interval: {config.analysis.collection_interval}s")
        print(f"   Report Interval: {config.analysis.report_interval}s")
        print(f"   Real-time Visualization: {'Yes' if config.analysis.enable_real_time_visualization else 'No'}")
        print(f"   Export Data: {'Yes' if config.analysis.export_data else 'No'}")
        
        print(f"\n📁 PATHS:")
        print(f"   Base Directory: {config.paths.base_directory}")
        print(f"   Results Directory: {config.paths.results_directory}")
        print(f"   Cleanup Temp Files: {'Yes' if config.paths.cleanup_temp_files else 'No'}")
        
        print(f"\n🔧 SYSTEM:")
        print(f"   Debug Mode: {'Yes' if config.debug_mode else 'No'}")
        print(f"   Verbose Logging: {'Yes' if config.verbose_logging else 'No'}")
        print(f"   Performance Monitoring: {'Yes' if config.performance_monitoring else 'No'}")


# Convenience functions for easy configuration access
def load_default_config() -> SystemConfig:
    """Load default configuration."""
    manager = ConfigManager()
    return manager.get_default_config()


def load_config_from_file(config_file: str) -> SystemConfig:
    """Load configuration from file."""
    manager = ConfigManager(config_file)
    return manager.load_config()


def create_config_for_scenario(scenario: str, environment: str = 'development') -> SystemConfig:
    """Create configuration for specific scenario and environment."""
    manager = ConfigManager()
    config = manager.get_default_config()
    
    # Apply scenario-specific settings
    scenario_config = manager.get_scenario_config(scenario)
    config.simulation.duration = scenario_config['duration']
    config.analysis.report_interval = scenario_config['report_interval']
    
    # Apply environment-specific settings
    env_config = manager.get_environment_config(environment)
    if 'simulation' in env_config:
        for key, value in env_config['simulation'].items():
            setattr(config.simulation, key, value)
    
    if 'analysis' in env_config:
        for key, value in env_config['analysis'].items():
            setattr(config.analysis, key, value)
    
    if 'system' in env_config:
        for key, value in env_config['system'].items():
            setattr(config, key, value)
    
    return config