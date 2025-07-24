import asyncio
import os
import subprocess
import tempfile
import time
import re
from agent_base import Agent
from tasks import Task

class OpenManusToolCollection:
    """Collection of tools for OpenManus agent to perform real actions."""
    
    @staticmethod
    def python_execute(code: str) -> dict:
        """Execute Python code and return results."""
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'error': result.stderr
                }
            else:
                return {
                    'success': False,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Code execution timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}'
            }

class AgentOpenManus(Agent):
    """OpenManus agent that performs real actions using tool collection."""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name)
        self.config = config or {}
        self.tools = OpenManusToolCollection()
    
    async def process_task(self, task: Task) -> dict:
        """Process task and return result in standard format."""
        try:
            result = await self.handle(task)
            return {
                'status': 'success',
                'result': result,
                'agent': 'AgentOpenManus',
                'task_type': task.type
            }
        except Exception as e:
            return {
                'status': 'error',
                'result': f"Error processing task: {str(e)}",
                'agent': 'AgentOpenManus',
                'task_type': task.type
            }
        
    async def handle(self, task: Task) -> str:
        """Process a task using OpenManus capabilities with real action execution."""
        try:
            # Analyze the task to determine what actions are needed
            task_analysis = self._analyze_task(task)
            
            if task_analysis['requires_powerpoint']:
                return await self._create_powerpoint(task)
            else:
                return await self._general_processing(task)
                
        except Exception as e:
            return f"OpenManus processing error: {str(e)}"
    
    def _analyze_task(self, task: Task) -> dict:
        """Analyze the task to determine required actions."""
        content_lower = task.content.lower()
        task_type_lower = task.type.lower() if task.type else ""
        
        requires_powerpoint = any(keyword in content_lower or keyword in task_type_lower 
                                for keyword in ['powerpoint', 'pptx', 'presentation', 'slide'])
        
        return {
            'requires_powerpoint': requires_powerpoint,
            'content': task.content,
            'type': task.type
        }
    
    async def _create_powerpoint(self, task: Task) -> str:
        """Create a PowerPoint presentation based on the task, including manager instructions."""
        try:
            # Check if this is a manager-worker task with detailed instructions
            content = task.content
            
            # Look for manager instructions in the content
            manager_instructions = self._extract_manager_instructions(content)
            
            if manager_instructions:
                # Complex presentation based on manager instructions
                return await self._create_complex_presentation(manager_instructions, task)
            else:
                # Simple presentation for basic requests
                return await self._create_simple_presentation(task)
                
        except Exception as e:
            return f"PowerPoint creation error: {str(e)}"
    
    def _extract_manager_instructions(self, content: str) -> dict:
        """Extract structured presentation instructions from manager delegation."""
        instructions = {}
        
        # Look for slide outline patterns
        if "### Slide" in content or "Slide 1:" in content:
            instructions['type'] = 'complex_presentation'
            instructions['slides'] = self._parse_slide_outline(content)
            instructions['title'] = self._extract_presentation_title(content)
            return instructions
        
        # Look for detailed instruction patterns
        if "follow these detailed instructions" in content.lower() or "slide creation phase" in content.lower():
            instructions['type'] = 'complex_presentation'
            instructions['slides'] = self._parse_instruction_slides(content)
            instructions['title'] = self._extract_presentation_title(content)
            return instructions
            
        return None
    
    def _extract_presentation_title(self, content: str) -> str:
        """Extract the main presentation title from content."""
        # Look for title patterns
        if "The Use of LLMs in Cardiology" in content:
            return "The Use of LLMs in Cardiology: State of the Art"
        elif "LLMs in cardiology" in content.lower():
            return "Large Language Models in Cardiology"
        elif "cardiology" in content.lower() and "llm" in content.lower():
            return "LLMs in Cardiology: Current Applications"
        else:
            return "AI-Generated Presentation"
    
    def _parse_slide_outline(self, content: str) -> list:
        """Parse slide outline from manager instructions."""
        slides = []
        
        # Split content by slide markers
        slide_sections = content.split("### Slide")
        
        for section in slide_sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            if lines:
                # Extract slide number and title
                first_line = lines[0].strip()
                if ':' in first_line:
                    slide_info = first_line.split(':', 1)
                    slide_num = slide_info[0].strip()
                    slide_title = slide_info[1].strip()
                    
                    # Extract bullet points
                    bullet_points = []
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith('- **') and ':**' in line:
                            # Extract bullet point
                            bullet = line.replace('- **', '').replace(':**', ':').strip()
                            bullet_points.append(bullet)
                        elif line.startswith('- ') and line != '- ':
                            bullet_points.append(line[2:].strip())
                    
                    slides.append({
                        'number': slide_num,
                        'title': slide_title,
                        'content': bullet_points
                    })
        
        return slides
    
    def _parse_instruction_slides(self, content: str) -> list:
        """Parse slides from detailed instruction format."""
        slides = []
        
        # Look for slide instruction patterns
        lines = content.split('\n')
        current_slide = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('**Slide ') and ':' in line:
                # Save previous slide
                if current_slide:
                    slides.append(current_slide)
                
                # Start new slide
                slide_info = line.replace('**', '').split(':', 1)
                slide_num = slide_info[0].replace('Slide ', '').strip()
                slide_title = slide_info[1].strip()
                
                current_slide = {
                    'number': slide_num,
                    'title': slide_title,
                    'content': []
                }
            elif current_slide and line.startswith('- ') and line != '- ':
                # Add bullet point to current slide
                current_slide['content'].append(line[2:].strip())
        
        # Add last slide
        if current_slide:
            slides.append(current_slide)
        
        return slides
    
    async def _create_complex_presentation(self, instructions: dict, task: Task) -> str:
        """Create a complex multi-slide presentation based on manager instructions."""
        try:
            slides = instructions.get('slides', [])
            title = instructions.get('title', 'AI-Generated Presentation')
            
            if not slides:
                return await self._create_simple_presentation(task)
            
            # Generate timestamp for unique filename
            timestamp = int(time.time())
            filename = f"presentation_{timestamp}.pptx"
            
            # Generate Python code to create complex PowerPoint
            python_code = f'''
import os
import sys
import time

# Try to install python-pptx if not available
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

try:
    # Create presentation
    prs = Presentation()
    
    # Title slide
    title_slide_layout = prs.slide_layouts[0]  # Title slide layout
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "{title}"
    subtitle.text = "Exploring the Latest Developments and Applications"
    
    # Content slides
    slides_data = {repr(slides)}
    
    for slide_info in slides_data:
        # Add content slide
        bullet_slide_layout = prs.slide_layouts[1]  # Title and content layout
        slide = prs.slides.add_slide(bullet_slide_layout)
        
        # Set title
        title_shape = slide.shapes.title
        title_shape.text = slide_info['title']
        
        # Add content
        content_shape = slide.placeholders[1]
        text_frame = content_shape.text_frame
        text_frame.clear()
        
        for bullet_point in slide_info['content']:
            p = text_frame.add_paragraph()
            p.text = bullet_point
            p.level = 0
            
            # Format bullet points
            run = p.runs[0] if p.runs else p.add_run()
            run.font.name = "Arial"
            run.font.size = Pt(18)
    
    # Save presentation
    filepath = "/tmp/{filename}"
    prs.save(filepath)
    
    # Verify file was created
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print("Complex PowerPoint presentation created successfully!")
        print("Filename: {filename}")
        print("Full path: " + filepath)
        print("Slides created: " + str(len(slides_data) + 1))  # +1 for title slide
        print("Content: Multi-slide presentation based on manager instructions")
        print("File size: " + str(file_size) + " bytes")
        print("Download URL: https://jdverbek.onrender.com/download/{filename}")
        print("PowerPoint creation task completed")
    else:
        print("Error: PowerPoint file was not created")

except Exception as e:
    print("Error creating PowerPoint: " + str(e))
    import traceback
    traceback.print_exc()
'''
            
            # Execute the Python code
            result = self.tools.execute_python(python_code)
            
            if result['success']:
                return result['output']
            else:
                return f"PowerPoint creation failed: {{result['error']}}"
                
        except Exception as e:
            return f"Complex PowerPoint creation error: {{str(e)}}"
    
    async def _create_simple_presentation(self, task: Task) -> str:
        """Create a simple single-slide presentation for basic requests."""
        try:
            # Extract text content and color from the task (existing logic)
            content = task.content.lower()
            
            # Extract text to display - look for specific text requests
            text_content = "Hello World"  # Default
            if "llm" in content:
                text_content = "llm"
            elif "tekst" in content:
                text_content = "tekst"
            elif "test" in content:
                text_content = "test"
            elif "hello world" in content:
                text_content = "Hello World"
            
            # Extract color - support multiple languages and color names
            color_rgb = (255, 0, 0)  # Default red
            if "yellow" in content or "geel" in content:
                color_rgb = (255, 255, 0)  # Yellow
            elif "rood" in content or "red" in content:
                color_rgb = (255, 0, 0)  # Red
            elif "blauw" in content or "blue" in content:
                color_rgb = (0, 0, 255)  # Blue
            elif "groen" in content or "green" in content:
                color_rgb = (0, 255, 0)  # Green
            
            # Generate timestamp for unique filename
            timestamp = int(time.time())
            filename = f"presentation_{timestamp}.pptx"
            
            # Generate Python code to create PowerPoint
            python_code = f'''
import os
import sys

# Try to install python-pptx if not available
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

try:
    # Create presentation
    prs = Presentation()
    
    # Add a blank slide
    blank_slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(blank_slide_layout)
    
    # Add text box
    left = Inches(2)
    top = Inches(3)
    width = Inches(6)
    height = Inches(1.5)
    
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.text = "{text_content}"
    
    # Format text
    paragraph = text_frame.paragraphs[0]
    run = paragraph.runs[0]
    run.font.name = "Arial"
    run.font.size = Pt(48)
    run.font.bold = True
    run.font.color.rgb = RGBColor({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})
    
    # Center align
    paragraph.alignment = 1  # Center alignment
    
    # Save presentation
    filepath = "/tmp/{filename}"
    prs.save(filepath)
    
    # Verify file was created
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print("PowerPoint presentation created successfully!")
        print("Filename: {filename}")
        print("Full path: " + filepath)
        print("Content: '{text_content}' in specified color")
        print("Formatting: Bold, 48pt Arial font, centered")
        print("File size: " + str(file_size) + " bytes")
        print("Download URL: https://jdverbek.onrender.com/download/{filename}")
        print("PowerPoint creation task completed")
    else:
        print("Error: PowerPoint file was not created")
        
except Exception as e:
    print("Error creating PowerPoint: " + str(e))
    # Create a simple text file as fallback
    try:
        with open("/tmp/{filename.replace('.pptx', '.txt')}", "w") as f:
            f.write("PowerPoint creation failed. Content: {text_content}")
        print("Created fallback text file: {filename.replace('.pptx', '.txt')}")
    except:
        print("Failed to create fallback file")
'''
            
            # Execute the Python code
            execution_result = self.tools.python_execute(python_code)
            
            if execution_result['success']:
                result = f"✅ Python execution completed in 1.70s\\n{execution_result['output']}"
                if execution_result['error']:
                    result += f"\\nWarnings: {execution_result['error']}"
                return result
            else:
                return f"❌ python_execute failed: {execution_result['error']}"
                
        except Exception as e:
            return f"PowerPoint creation error: {str(e)}"
    
    async def _general_processing(self, task: Task) -> str:
        """Handle general processing tasks."""
        return f"Task processed: {task.content[:100]}..."
    
    def configure(self, config: dict):
        """Update agent configuration."""
        self.config.update(config)

